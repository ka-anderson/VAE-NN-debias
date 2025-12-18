import re
import pandas as pd

def read_pd_markdown(path):
    # https://stackoverflow.com/questions/60154404/is-there-the-equivalent-of-to-markdown-to-read-data
    df = pd.read_table(path, sep="|", header=0, index_col=1, skipinitialspace=True).dropna(axis=1, how='all').iloc[1:] 
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.lstrip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if not converted.isna().any():
            df[col] = converted

    # df = df.apply(pd.to_numeric, errors='ignore')

    return df


def torch_to_numpy_image(torch_image):
    '''
    move an image to the cpu and switch the channel dim, so that we get the default numpy structure (w, h, c)
    '''
    if len(torch_image.shape) == 3:
        return torch_image.detach().cpu().permute(1, 2, 0)
    else:
        return torch_image.detach().cpu().permute(0, 2, 3, 1)
    
def get_number(string_input):
    number = re.findall(r"-?\d+(?:\.\d+)?", string_input)[0] # https://stackoverflow.com/questions/73580598/extracting-float-or-int-number-and-substring-from-a-string

    if "." in number:
        return float(number)
    return int(number)