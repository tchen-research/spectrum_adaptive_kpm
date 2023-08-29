import subprocess

files = [
    'fig1_support',
    'fig2_moments',
    'fig3_zincblende',
    'fig4_parsec',
]

for file in files:
    subprocess.run(f'jupyter nbconvert --execute --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 {file}.ipynb',shell=True)

