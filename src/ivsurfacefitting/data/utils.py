import pandas as pd
import torch


def df_to_tensor(
    indeces: pd.Index, valuescols: list[str], df: pd.DataFrame
) -> torch.Tensor:
    """
    Converts a data frame with columns valuescols into a tensor.

    Args:
        indeces (pd.Series): Indeces to be converted to tensor.
        valuescols (list[str]): Columns to transform.
        df (pd.DataFrame): Dataframe to convert.

    Returns:
        torch.Tensor
    """
    unique_indeces = indeces.drop_duplicates()
    for col in valuescols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not in the dataframe.")

    filtered_df = df[df.index.isin(indeces)]

    counts = filtered_df.index.value_counts()
    if len(counts.drop_duplicates()) != 1:
           raise ValueError("All indices must have the same amount of rows.")
    n_points = counts.iloc[0]

    ordered = filtered_df.loc[unique_indeces,valuescols]

    values = ordered.to_numpy(dtype = "float32")

    tensor = torch.from_numpy(values).reshape(
        len(unique_indeces), n_points, len(valuescols)
    )

    return tensor


def tensor_to_df(
    indeces: pd.Index, valuescols: list[str], tensor: torch.Tensor
) -> pd.DataFrame:
    """
    Converts tensor into a dataframe with given columns and indices.

    Tensor must have shape (number of indices, rows per index, len of valuescols).

    Args:
        indexcol (str): Name of indexing column.
        indeces (pd.Series): Indeces to be converted to tensor.
        valuescols (list[str]): Columns to transform.
        tensor (torch.Tensor): Tensor to convert.

    Returns:
        pd.DataFrame
    """
    unique_indeces = indeces.drop_duplicates()
    n_indices, n_rows, n_values = tensor.shape

    if n_indices != len(unique_indeces):
        raise ValueError(
            f"Tensor has {n_indices} indices, but {len(indeces)} indices were provided."
        )

    if n_values != len(valuescols):
        raise ValueError(
            f"Tensor has {n_values} value columns, but {len(valuescols)} were provided."
        )

    values = tensor.detach().cpu().numpy().reshape(-1, n_values)
    df = pd.DataFrame(values, columns=valuescols)
    df.index = indeces

    return df
