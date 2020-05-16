# Mind the Gap: Bit-vector Interpolation recast over Linear Integer Arithmetic
## What's this?
An implementation of [1] submitted to the artifact evaluation for TACAS20 (https://tacas.info/artifacts-20.php).  
You can get the environment of the artifact evaluation here: https://figshare.com/articles/tacas20ae_ova/9699839
The necessary libraries are shown in `Pipfile`s in each directory.

Abstract:
> Much of an interpolation engine for bit-vector (BV) arithmetic can be constructed by observing that BV arithmetic can be modeled with linear integer arithmetic (LIA). Two BV formulae can thus be translated into two LIA formulae and then an interpolation engine for LIA used to derive an interpolant, albeit one expressed in LIA. The construction is completed by back-translating the LIA interpolant into a BV formula whose models coincide with those of the LIA interpolant. This paper develops a back-translation algorithm showing, for the first time, how back-translation can be universally applied, whatever the LIA interpolant. This avoids the need for deriving a BV interpolant by bit-blasting the BV formulae, as a backup process when back-translation fails. The new back-translation process relies on a novel geometric technique, called gapping, the correctness and practicality of which are demonstrated.

## Structure
- experiment: main code
    - small.txt: the benchmark run in the experiment
- make_figure: Jupyter notebook to generate figures in the paper
- mathsat-5.5.4-linux-x86_64: MathSAT5.  The content is omitted.  
- Readme_TACAS20_AE.txt: Readme.txt submitted to the TACAS20 Artifact Evaluation comittee.  You can replicate the result by gathering third party resources (MathSAT5 and benchmark programs) and following this.

## Omitted files
- experiment/samples/nfm2017/...: the directory from http://forsyte.at/softw are/demy/nfm17.tar.gz [2]
- mathsat-5.5.4-linux-x86_64/...: You can get it from https://mathsat.fbk.eu/download.html .

## Author
Takamasa Okudono (http://group-mmm.org/~tokudono/), Andy King


- [1] Takamasa Okudono, Andy King: Mind the Gap: Bit-vector Interpolation recast over Linear Integer Arithmetic. TACAS (1) 2020: 79-96
- [2] Yulia Demyanova, Philipp Rümmer, Florian Zuleger: Systematic Predicate Abstraction Using Variable Roles. NFM 2017: 265-281