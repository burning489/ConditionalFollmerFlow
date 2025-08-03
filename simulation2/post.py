import os
import re

import pandas as pd


## convergence w.r.t. number of training samples

rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
duration_pattern = re.compile(r'avg. duration \(repeated \d+ times\): ([\d.]+) sec.')
metric1_pattern = re.compile(r'MSE\(mean\) distance: ([\d.]+)')
metric2_pattern = re.compile(r'MSE\(std\) distance: ([\d.]+)')
datasets = ['m1', 'm2', 'm3']

rows = []
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    n = int(n)
    T = float(T) if T else None
    if mode != 'follmer' or T is not None or dataset not in datasets:
        continue
    logfile = os.path.join('logs', rundir, 'test.log')
    if not os.path.exists(logfile):
        continue
    with open(logfile, 'r') as f:
        row = {'dataset': dataset, 'n': n, 'mode': mode, 'T': T}
        for line in f:
            if match := duration_pattern.search(line):
                duration = float(match.group(1))
                row['GPU-T'] = duration
            if match := metric1_pattern.search(line):
                mean_mse = float(match.group(1))
                row['MSE_1'] = mean_mse
            if match := metric2_pattern.search(line):
                std_mse = float(match.group(1))
                row['MSE_2'] = std_mse
    rows.append(row)


df = pd.DataFrame(rows, columns=['n', 'dataset', 'mode', 'T', 'MSE_1', 'MSE_2', 'GPU-T'])
df.sort_values(by=['dataset', 'n'], ascending=[True, True], inplace=True)
df.rename(columns={'n': '$n$', 'T': '$T$'}, inplace=True)
# print(df)



pivot_df = df.pivot(index='$n$', columns='dataset', values=['MSE_1', 'MSE_2']) # pyright: ignore[reportArgumentType]
# ('AVE', '4squares') -> ('4squares', 'AVE')
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$n$', float_format='%.3f', column_format='ccccccccccc', multicolumn_format='c', # pyright: ignore[reportArgumentType]
                        caption=r'In simulation study II, influence of $n$ on the ODE-based method with F\"ollmer coefficient.'))


## influence of stopping time

rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
duration_pattern = re.compile(r'avg. duration \(repeated \d+ times\): ([\d.]+) sec.')
metric1_pattern = re.compile(r'MSE\(mean\) distance: ([\d.]+)')
metric2_pattern = re.compile(r'MSE\(std\) distance: ([\d.]+)')
datasets = ['m1', 'm2', 'm3']
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
    if mode != 'follmer' or n != 10000 or T not in T_list or dataset not in datasets:
        continue
    logfile = os.path.join('logs', rundir, 'test.log')
    if not os.path.exists(logfile):
        continue
    with open(logfile, 'r') as f:
        row = {'dataset': dataset, 'n': n, 'mode': mode, 'T': T}
        for line in f:
            if match := duration_pattern.search(line):
                duration = float(match.group(1))
                row['GPU-T'] = duration
            if match := metric1_pattern.search(line):
                mean_mse = float(match.group(1))
                row['MSE_1'] = mean_mse
            if match := metric2_pattern.search(line):
                std_mse = float(match.group(1))
                row['MSE_2'] = std_mse
    rows.append(row)



df = pd.DataFrame(rows, columns=['dataset', 'T', 'MSE_1', 'MSE_2', 'GPU-T', 'n', 'mode'])
df.sort_values(by=['dataset', 'T'], ascending=[True, True], inplace=True)
df['T'] = df['T'].apply(lambda x: format(x, 'g'))
df.rename(columns={'n': '$n$', 'T': '$T$'}, inplace=True)
# print(df)



pivot_df = df.pivot(index='$T$', columns='dataset', values=['MSE_1', 'MSE_2']) # pyright: ignore[reportArgumentType]
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$T$', float_format='%.3f', column_format='ccccccccccc', multicolumn_format='c', # pyright: ignore[reportArgumentType]
                        caption=r'In simulation study II, influence of stopping time $T$ on the sample average and standard deviation of 5000 samples obtained from the ODE-based method with F\"ollmer coefficient.'))


## comparison with VAE, GAN, linear flow and VE-SDE


rundir_pattern = re.compile(r'(\w+)-n(\d+)-(\w+)(?:-T([\d.]+))?')
duration_pattern = re.compile(r'avg. duration \(repeated \d+ times\): ([\d.]+) sec.')
metric1_pattern = re.compile(r'MSE\(mean\) distance: ([\d.]+)')
metric2_pattern = re.compile(r'MSE\(std\) distance: ([\d.]+)')
datasets = ['m1', 'm2', 'm3']

rows = []
for rundir in os.listdir('logs'):
    match = rundir_pattern.search(rundir)
    if not match:
        print(f'{rundir} does not match rundir_pattern')
        continue
    dataset, n, mode, T = match.groups()
    T = float(T) if T else None
    n = int(n)
    if (mode == 'follmer' and T is not None or n != 10000) or dataset not in datasets:
        continue
    logfile = os.path.join('logs', rundir, 'test.log')
    if not os.path.exists(logfile):
        continue
    with open(logfile, 'r') as f:
        row = {'dataset': dataset, 'mode': mode}
        for line in f:
            if match := duration_pattern.search(line):
                duration = float(match.group(1))
                row['GPU-T'] = duration
            if match := metric1_pattern.search(line):
                mean_mse = float(match.group(1))
                row['MSE_1'] = mean_mse
            if match := metric2_pattern.search(line):
                std_mse = float(match.group(1))
                row['MSE_2'] = std_mse
    rows.append(row)



df = pd.DataFrame(rows, columns=['dataset', 'MSE_1', 'MSE_2', 'GPU-T', 'mode'])
df['mode'] = df['mode'].map({
    'follmer': r'F\"ollmer',
    'vae': 'VAE',
    'gan': 'GAN',
    'linear': 'Linear',
    'trig': 'Trigonometric',
    'vesde': 'VE-SDE',
    'nnkcde': 'NNKCDE',
    'flexcode': 'FlexCode',
})
df.sort_values(by=['dataset', 'mode'], ascending=[True, True], inplace=True)


pivot_df = df.pivot(index='mode', columns='dataset', values=['MSE_1', 'MSE_2']).rename_axis('method') # pyright: ignore[reportArgumentType]
pivot_df.columns = pivot_df.columns.swaplevel(0, 1) # pyright: ignore[reportAttributeAccessIssue]
pivot_df.columns.names = [None, None]
pivot_df.sort_index(axis=1, level=0, inplace=True)
print(pivot_df)


print(pivot_df.to_latex(index='$T$', float_format='%.3f', column_format='ccccccccccc', multicolumn_format='c'))


