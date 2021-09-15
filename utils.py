import re
import pandas as pd
import unidecode

# Regex to identify common markdown in a string
MATCH_MARKDOWN = re.compile('.*(\\*\\*|__|~~|\\]\\(http://).*')


def has_markdown(s: str):
    """Identify if a string contains common markdown based on a regular expression."""
    if isinstance(s, str):
        m = MATCH_MARKDOWN.match(s)
        return m is not None
    else:
        return False


def hour_min_sec(secs: int, hms=False):
    """Convert seconds into a more readable hh:mm:ss representation
    :secs Number of seconds
    :hms Hours:Minutes:Seconds representation, rather than the default seconds."""
    if secs is not None:
        m, s = divmod(secs, 60)
        m = int(m)
        s = int(s)
        if m > 59:
            h, m = divmod(m, 60)
            h = int(h)
            m = int(m)
            if hms:
                return f'{h}h {m:02d}m {s:02d}s'
            else:
                return f'{h}:{m:02d}:{s:02d}'
        else:
            if hms:
                return f'{m}m {s:02d}s'
            else:
                return f'{m}:{s:02d}'
    else:
        return '_:__'


def trunc_length(df: pd.DataFrame, max_length: int):
    """Remove items where length of comment field in dataframe `df` larger than `max_length`

    Deprecated!
    """
    in_rows = df.shape[0]
    df['comment_length'] = df['comment'].str.len()
    df['parent_comment_length'] = df['parent_comment'].str.len()
    df = df[df['comment_length'] <= max_length]
    out_rows = df.shape[0]
    print(f"trunc_length(df, {max_length}): removed {in_rows-out_rows} out of {in_rows} to new total of {out_rows}")
    return df


def clean_string(df: pd.DataFrame, field='comment'):
    """Truncate length of comment field in dataframe `df` to `max_length`

    Deprecated!
    """
    in_rows = df.shape[0]
    df = df.dropna(subset=[field])
    df[field] = df[field].apply(unidecode.unidecode)
    out_rows = df.shape[0]
    print(f"clean_string(df, field={field}): removed {in_rows-out_rows} out of {in_rows} to new total of {out_rows}")
    return df


def combine_with_context(df: pd.DataFrame):
    """Combine `comment` and `parent_comment` fields in a dataframe to a single string in field `target`"""
    df['target'] = df['parent_comment'] + ' ' + df['comment']
    return df
