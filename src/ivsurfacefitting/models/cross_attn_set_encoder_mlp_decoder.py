from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from src.ivsurfacefitting.data.utils import df_to_tensor, tensor_to_df
from src.ivsurfacefitting.experiments.evaluation import (
    IVSurfaceEvalConfig,
    IVSurfaceEvalResults,
)
from src.ivsurfacefitting.experiments.train import (
    IVSurfaceTrainConfig,
    IVSurfaceTrainResults,
)
from src.ivsurfacefitting.models.base import IVSurfaceModel


class CrossAttnEncodeMLPDecoder(IVSurfaceModel, nn.Module):
    """
    Transformer model for fitting gridless surfaces.

    The idea is that, given a set of observations X = {x_1, x_2, ...}, the decoder will use a
    set of learneable vectors Z = {z_1,..}, such that the first attention matrix is

        A = softmax(QK^T / sqrt(d))

    Where Q comes from WQ*Z and K from WK*X, then doing V*A whete V = WV*X, ensures
    permutation invariance, since it would permute the rows of both K and V.

    After this we perform more transofrmer layers for the encoding, finally one
    uses a fully connected neural network for the decoder.

    Note that it doesnt guarantee nor try to enforce no arbitrage in any way.
    """

    def __init__(
        self,
        name: str = "CrossAttnEncodeMLPDecoder",
        latent_dim: int = 16,
        learneable_dim=32,
    ) -> None:
        """
        Initializes the module and defines the architecture.

        Remember that it takes log moneyness and maturity as inputs,
        and outputs iv, so the input dimension is 2 and output is one.

        Args:
            name (str): Name of the model.
            latent_dim (int): Dimension to encode into.
            learneable_dim (int): Dimension of learneable vectors.
        """

        IVSurfaceModel.__init__(self, name=name, learnable=True)
        nn.Module.__init__(self)

        # Embedding of initial observations
        # After this shape is (batch, N, 64)
        self.embedding = nn.Sequential(
            nn.Linear(3, 64),  # 3 is input dim plus output dim
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Create learneable vectors Z
        self.learneable = nn.Parameter(torch.randn(learneable_dim, 64))

        # Create permutation invariant cross-attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=8,
            batch_first=True,
        )

        # Make multiple transformer layer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=8,
            dim_feedforward=128,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=5,
        )

        # Final linear layers to get low dimensional representation.

        self.final_encoding_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(learneable_dim * 64, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

        # Decoding

        self.network = nn.Sequential(
            nn.Linear(latent_dim + 2, 256),  # latent dim + input dim
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),  # output dim
        )

    def forward(self, samples: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Encodes the samples, and then evaluates decoder on the given coordinates.

        Args:
            samples (torch.Tensor): samples used for fitting
            coordinates (torch.Tensor): coordinates at which to evaluate the model.
        """
        # Encoding
        samples = self.embedding(samples)

        # Copy latent vectors by batch
        batch_size = samples.shape[0]
        latents = self.learneable.unsqueeze(0)
        latents = latents.expand(batch_size, -1, -1)

        # Perform attention, returns weights too, which are usefull sometimes, not currently needed
        x, _ = self.cross_attention(
            query=latents,
            key=samples,
            value=samples,
        )

        # Now transformer layers
        x = self.transformer(x)

        # FInal layer
        encoding = self.final_encoding_layer(x)

        # Decoding
        _, N, _ = coordinates.shape

        # Turn latent into shape (batch, points, latent_dim) so it can be concatenated with coordinates.
        z = encoding[:, None, :].expand(-1, N, -1)

        x = torch.cat([z, coordinates], dim=-1)

        return self.network(x)

    def reset_parameters(self):
        """
        Resets the parameters of all layers.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters") and not module is self:
                module.reset_parameters()  # Not sure how to fix this pyright issue.

    def learn(self, train_config: IVSurfaceTrainConfig) -> IVSurfaceTrainResults:
        """
        Handles the learning/training.

        Note that sue to the nature of the transformer, each batch is forced to have the exavt same input size,
        so it will raise an error if the sample sizes are different, this is unavoideable for this architecture,
        thus either fix the input size of the datasets, or lose information.

        Args:
            train_data (pd.DataFrame)
        """
        self.reset_parameters()
        train_data = train_config.getdata()

        train_tensor = df_to_tensor(
            train_data.index,
            ["logmoneyness", "maturity", "iv"],
            train_data,
        )

        input_dim = 2
        output_dim = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        criterion = torch.nn.MSELoss()
        n_surfaces, n_points, dimensions = train_tensor.shape

        surface_indeces = np.arange(n_surfaces)

        batch_size = 64

        if dimensions != input_dim + output_dim:
            raise ValueError("Tensor dimension doesnt match.")

        for _ in tqdm(range(50)):
            np.random.shuffle(surface_indeces)

            for batch in range(n_surfaces // batch_size):
                p = np.random.randint(5, n_points)

                start_index = batch * batch_size

                batch_indices = surface_indeces[start_index : start_index + batch_size]

                batch_data = train_tensor[batch_indices].to(device)

                point_indices = np.random.choice(
                    np.arange(n_points), size=p, replace=False
                )

                samples = batch_data[:, point_indices, :]

                coordinates = batch_data[:, :, :input_dim]

                values = batch_data[:, :, input_dim:]

                predictions = self(samples, coordinates)

                loss = criterion(predictions, values)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

        self.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.eval()

        # TODO: Training results

        return IVSurfaceTrainResults()

    def fit(self, eval_config: IVSurfaceEvalConfig) -> IVSurfaceEvalResults:
        """
        Fits the results.
        """

        test, context, grid = eval_config.getdata()

        grid.insert(0, "id", 0)
        grid = grid.set_index("id")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        with torch.no_grad():
            # cast to fix pyright errors
            test_coordinates = cast(
                pd.DataFrame, test[[ "logmoneyness", "maturity"]]
            )
            indices = test.index

            # Test evaluating
            context_tensor = df_to_tensor(
                indices,
                ["logmoneyness", "maturity", "iv"],
                context,
            ).to(device)

            test_coords_tensor = df_to_tensor(
                indices,
                ["logmoneyness", "maturity"],
                test_coordinates,
            ).to(device)

            results_tensor = self.forward(context_tensor, test_coords_tensor).to("cpu")

            ivs_results = tensor_to_df(
                indices,
                ["iv"],
                results_tensor,
            )

            test_results = test_coordinates.copy()

            test_results["iv"] = ivs_results["iv"]

            # Grid evaluating
            grid_tensor = df_to_tensor(
                grid.index,
                ["logmoneyness", "maturity"],
                grid,
            ).to(device)

            grid_tensor = grid_tensor.repeat(len(indices.unique()), 1, 1)

            results_tensor = self.forward(context_tensor, grid_tensor).to("cpu")

            unique_ids = indices.drop_duplicates()
            grid_full_index = unique_ids.repeat(len(grid))

            ivs_results = tensor_to_df(
                grid_full_index,
                ["iv"],
                results_tensor,
            )

            grid_results = pd.concat([grid]*len(unique_ids), ignore_index=True)
            grid_results.index = grid_full_index
            grid_results["iv"] = ivs_results["iv"].to_numpy()

            surface_info = pd.DataFrame(indices.drop_duplicates(), columns=["id"]).set_index("id") # No surface info to store (yet?)

        self.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return IVSurfaceEvalResults(test_results, grid_results, surface_info)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
