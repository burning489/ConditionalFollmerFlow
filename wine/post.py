import os
import re

import pandas as pd


## convergence w.r.t. number of training samples

rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
metric_pattern = re.compile(r'Coverage distance: ([\d.]+)')

rows = []
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    n = int(n)
    T = float(T) if T else None
    if mode != 'follmer' or T is not None:
        continue
    for logfile in os.listdir(os.path.join('logs', rundir)):
        match = re.search(r'test-alpha([\d.]+)\.log', logfile)
        if not match:
            continue
        alpha = float(match.group(1))
        with open(os.path.join('logs', rundir, logfile), 'r') as f:
            row = {'dataset': dataset, 'n': n, 'mode': mode, 'T': T, 'alpha': alpha}
            for line in f:
                if match := metric_pattern.search(line):
                    mean_mse = float(match.group(1))
                    row['coverage'] = mean_mse
        rows.append(row)


df = pd.DataFrame(rows, columns=['n', 'mode', 'T', 'coverage', 'alpha'])
df.sort_values(by=['n',], ascending=[True,], inplace=True)
df.rename(columns={'n': '$n$', 'T': '$T$'}, inplace=True)


pivot_df = df.pivot(index='$n$', columns='alpha', values=['coverage']) # pyright: ignore[reportArgumentType]
# ('AVE', '4squares') -> ('4squares', 'AVE')
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$n$', float_format='%.4f', column_format='ccccccccccc', multicolumn_format='c', # pyright: ignore[reportArgumentType]
                        caption=r'In Real Data Analysis I, influence of $n$ on the ODE-based method with F\"ollmer coefficient.'))


## influence of stopping time

rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
metric_pattern = re.compile(r'Coverage distance: ([\d.]+)')
T_list = [0.999, 0.9995, 0.9999]

rows = []
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    n = int(n)
    T = float(T) if T else None
    if mode != 'follmer' or n != 5847 or T not in T_list:
        continue
    for logfile in os.listdir(os.path.join('logs', rundir)):
        match = re.search(r'test-alpha([\d.]+)\.log', logfile)
        if not match:
            continue
        alpha = float(match.group(1))
        with open(os.path.join('logs', rundir, logfile), 'r') as f:
            row = {'dataset': dataset, 'n': n, 'mode': mode, 'T': T, 'alpha': alpha}
            for line in f:
                if match := metric_pattern.search(line):
                    std_mse = float(match.group(1))
                    row['coverage'] = std_mse
        rows.append(row)


df = pd.DataFrame(rows, columns=['T', 'coverage', 'n', 'mode', 'alpha'])
df.sort_values(by=['T',], ascending=[True,], inplace=True)
df['T'] = df['T'].apply(lambda x: format(x, 'g'))
df.rename(columns={'n': '$n$', 'T': '$T$'}, inplace=True)


pivot_df = df.pivot(index='$T$', columns='alpha', values=['coverage']) # pyright: ignore[reportArgumentType]
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$T$', float_format='%.4f', column_format='ccccccccccc', multicolumn_format='c', # pyright: ignore[reportArgumentType]
                        caption=r'In Real Data Analysis I, influence of stopping time $T$ on the sample average and standard deviation of 5000 samples obtained from the ODE-based method with F\"ollmer coefficient.'))


## comparison with VAE, GAN, linear flow and VE-SDE


rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
metric_pattern = re.compile(r'Coverage distance: ([\d.]+)')

rows = []
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    T = float(T) if T else None
    n = int(n)
    if mode == 'follmer' and (T is not None or n != 5847):
        continue
    for logfile in os.listdir(os.path.join('logs', rundir)):
        match = re.search(r'test-alpha([\d.]+)\.log', logfile)
        if not match:
            continue
        alpha = float(match.group(1))
        with open(os.path.join('logs', rundir, logfile), 'r') as f:
            row = {'dataset': dataset, 'mode': mode, 'alpha': alpha}
            for line in f:
                if match := metric_pattern.search(line):
                    std_mse = float(match.group(1))
                    row['coverage'] = std_mse
        rows.append(row)



df = pd.DataFrame(rows, columns=['coverage', 'mode', 'alpha'])
df['mode'] = df['mode'].map({
    'follmer': r'F\"ollmer',
    'vae': 'VAE',
    'gan': 'GAN',
    'linear': 'Linear',
    'trig': 'Trigonometric',
    'vesde': 'VE-SDE',
})
df.sort_values(by=['mode',], ascending=[True,], inplace=True)
# print(df)



pivot_df = df.pivot(index='mode', columns='alpha', values=['coverage']).rename_axis('method') # pyright: ignore[reportArgumentType]
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$T$', float_format='%.4f', column_format='ccccccccccc', multicolumn_format='c'))


