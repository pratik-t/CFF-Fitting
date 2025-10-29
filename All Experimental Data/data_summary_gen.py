"""
This script generates a Markdown file that summarizes in a
human-readable fashion all of the `.csv` files that we have
in the directory `data`.
"""

# Native Library | os:
import os

# Native Library | os:
import glob

# Native Library | os:
import re

# 3rd Part Library | Pandas:
import pandas as pd

# (1): Define static string to set current directory:
script_dir = os.path.dirname(os.path.abspath(__file__))

# (2): Define static string to help script find its entry point:
DATA_FOLDER_PATH = os.path.abspath(os.path.join(script_dir, '..', 'All Data'))

# (3): Define static string that will be the *name* of the output `.md`:
OUTPUT_MARKDOWN_FILE_NAME = 'README.md'

def get_non_empty_columns(df: pd.DataFrame) -> list:
    """
    ## Description:
    Return a list of column names in a Pandas DataFrame that have at least 
    one non-empty value, excluding 'link' (which is a string of a URL) and
    any systematic or statistical uncertainty columns.
    """
    non_empty = []
    exclude_pattern = re.compile(r"(sig_BSA|del_ALU|_sys_plus|_sys_minus|_stat_plus|_stat_minus|_min|_max)$")
    for col in df.columns:
        if col == 'link' or exclude_pattern.search(col):
            continue
        series = df[col].dropna().astype(str).str.strip()
        if series.ne('').any():
            non_empty.append(col)
    return non_empty


def format_header(name: str) -> str:
    """
    ## Description:
    Map column names to LaTeX-style labels as specified.
    """
    if name == 'q_squared':
        return r'$Q^{2}$'
    if name == 'x_b':
        return r'$x_{b}$'
    if name == 'w':
        return r'$W$'
    if name in ('k', 't', 'BSA', 'TSA', 'DSA', 'BCA', 'ALU'):
        return f'${{{name}}}$'
    if name == 'Helc_charge_asym':
        return r'$Q_{asy}$'
    if name == 'phi':
        return r'$\phi$'
    if name == 'ALU_sin_PHI':
        return r'$A_{LU}^{\sin\phi}$'
    if name == 'ALU_sin_2PHI':
        return r'$A_{LU}^{\sin\ 2\phi}$'
    if name == 'cAUT':
        return r'$c_{A_{UT}}$'
    if name == 'cALT':
        return r'$c_{A_{LT}}$'
    if name == 'cos_theta*gamma_gamma':
        return r'$\cos\theta^{*}_{\gamma\gamma}$'
    if name == 'cos_theta_CM':
        return r'$\cos\theta_{c.m.}$'
    if name == 'sigma [nb]':
        return r'$\sigma\  [nb]$'
    if name == 'dsigma/dt [nb/GeV^2]':
        return r'$d\sigma/dt\  [nb/GeV^{2}]$'
    if name == 'D5_sigma (fb/(MeV sr2))':
        return r'$d^{5}\sigma/(dk_{lab}\ d\Omega_{e_{lab}}\ d\Omega_{p_{c.m.}})\ [fb/(MeV\ sr^{2})]$'
    if name == 'D2_sigma_d_omega (nb/sr)':
        return r'$d^{2}\sigma/d\Omega_{p_{c.m.}}\  [nb/sr]$'
    if name == 'D^4_sigma (nb/Gev^4)':
        return r'$d^{4}\sigma/(dQ^{2}\ dt\ dx_{B}\ d\phi)\ [nb/GeV^{4}]$'
    if name == 'Helc_diff_D^4_sigma (nb/Gev^4)':
        return r'$(d^{4}\sigma^{+}-d^{4}\sigma^{-})\ [nb/GeV^{4}]$'
    if name == '1/2 Helc_sum_d4_sigma (nb/GeV^4)':
        return r'$1/2\times(d^{4}\sigma^{+}+d^{4}\sigma^{-})\ [nb/GeV^{4}]$'
    if name == '1/2 Helc_diff_d4_sigma (nb/GeV^4)':
        return r'$1/2\times(d^{4}\sigma^{+}-d^{4}\sigma^{-})\ [nb/GeV^{4}]$'
    if name == 'theta_e':
        return r'$\theta_{e}$'
    if name == 'theta_q':
        return r'$\theta_{q}$'
    if name == 'E_gamma':
        return r'$E_{\gamma}$'
    if name == 'chi^2_pol_by_dof':
        return r'$\chi^{2}_{pol}/dof$'
    if name == 'chi^2_unpol_by_dof':
        return r'$\chi^{2}_{unpol}/dof$'
    if name == 'Slope [GeV^-2]':
        return r'$slope\ [/GeV^{2}]$'
    return name


def process_csv_folder(folder_path: str, output_md: str):
    """
    ## Description:
    Scan folder_path for CSVs, group them by their first 'link' entry, and write a Markdown.
    """
    link_groups = {}

    for filepath in glob.glob(os.path.join(folder_path, '*.csv')):
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)
        non_empty_cols = get_non_empty_columns(df)
        first_link = ''
        if 'link' in df.columns:
            links = df['link'].dropna().astype(str).str.strip().tolist()
            if links:
                first_link = links[0]
        link_groups.setdefault(first_link, []).append((filename, non_empty_cols))

    file_path = f"{DATA_FOLDER_PATH}/{output_md}"

    with open(file_path, 'w', encoding='utf-8') as md:
        md.write('# Data Summary\n\n')
        for link, file_list in link_groups.items():
            md.write(f'- Link: {link}\n')
            for filename, cols in file_list:
                md.write(f'  - {filename} :\n')
                if cols:
                    formatted = [format_header(c) for c in cols]
                    header_row = '| ' + ' | '.join(formatted) + ' |'
                    separator = '| ' + ' | '.join(['---'] * len(formatted)) + ' |'
                    md.write(f'    {header_row}\n')
                    md.write(f'    {separator}\n')
                else:
                    md.write('    _No non-empty columns found._\n')
            md.write('\n')


if __name__ == '__main__':
    process_csv_folder(DATA_FOLDER_PATH, OUTPUT_MARKDOWN_FILE_NAME)
