# Instruction to replicate the table and graph in Experiment section
Thank you for evaluating our artifact.
Please follow the step-by-step introduction to replicate the result.
Please put `mind_the_gap` directory in the home directory, and we assume that 
the current directory is now `/home/tacas20ae`.
If the password is required in the steps, please type "tacas20ae".

1. Install required packages in Ubuntu (If it fails, the startup updating can be running.  
Please wait for about 5 minutes and try it later):
```
sudo apt update
sudo apt install libffi-dev libsqlite3-dev libbz2-dev libncurses5-dev \
libgdbm-dev liblzma-dev libssl-dev tcl-dev tk-dev libreadline-dev \
python-pip build-essential cmake mercurial cython libgmp3-dev graphviz
```

2. Install pyenv
```
curl https://pyenv.run | bash
```

3. Modify PATH to use pyenv and MathSAT.
Please add following lines to the end of `~/.bash_profile`
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
PYTHON_BIN_PATH="$(python3 -m site --user-base)/bin"
PATH="$PATH:$PYTHON_BIN_PATH"
export LD_LIBRARY_PATH=/home/tacas20ae/mind_the_gap/mathsat-5.5.4-linux-x86_64/lib
```

4. Load .bash_profile
```
source ~/.bash_profile
```

5. Install Python 3.7.3
```
yes N | pyenv install 3.7.3
pyenv global 3.7.3
```

6. Install pipenv
```
pip install pipenv --user
```

7. Change current directory to `~/mind_the_gap/mathsat-5.5.4-linux-x86_64/python`
```
cd ~/mind_the_gap/mathsat-5.5.4-linux-x86_64/python
```

8. Build MathSAT5 and setup
```
pipenv run setup.py build
cp ~/mind_the_gap/mathsat-5.5.4-linux-x86_64/python/build/lib.linux-x86_64-3.7/* ~/mind_the_gap/experiment
cp ~/mind_the_gap/mathsat-5.5.4-linux-x86_64/lib/*.so ~/mind_the_gap/experiment
cp ~/mind_the_gap/mathsat-5.5.4-linux-x86_64/python/*.py ~/mind_the_gap/experiment
```

9. Change current directory to experiment directory
```
cd ~/mind_the_gap/experiment
```

10. (If needed) Change the timeout.  There are 936 tasks and the timeout is currently set to 600 seconds.
It means the experiment can take about one week at worst.  To shrink the timeout, please change Line 82 of `~/mind_the_gap/experiment/oden.py`.
For example, to change the timeout to 60 seconds (whole the experiment would finish in 4 hours), change the line like:
```
    to = 60 #temp
```

11. Run the experiment
```
pipenv run oden.py local
```

12. Make the summary of the experiment
```
pipenv run make_summary
```

13. Copy the summary
```
cp ~/mind_the_gap/experiment/summary/summary.csv ~/mind_the_gap/make_figure/summary_1023_2.csv
```

14. Change the current directory
```
cd ~/mind_the_gap/make_figure
```

14. Run the script to make figures and tables
```
pipenv run jupyter nbconvert --to notebook --execute Untitled.ipynb
```

15. Check the result
- mean.csv: "Time" and "Size" in the left table on Table 1
- count.csv: "Solved" column of the left table on Table 1
- cross.csv: the right table on Table 1
- scatter_naive_boxing.pdf: the left scatter plot of Figure 5
- line_runtime_naive_boxing.pdf: the right line graph of Figure 5
- line_size_interpolant_boxinv_naive.pdf: the left line graph of Figure 6
- scatter_time_size_naive_boxing.pdf: the right scatter plot of Figure 6


Thank you very much!
