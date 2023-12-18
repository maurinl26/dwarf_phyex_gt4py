cd $HOME/dwarf_phyex_gt4py
source .venv/bin/activate

export PYTHONPATH=$PYTHONPATH:$HOME/dwarf_phyex_gt4py/src

python src/phyex_gt4py/drivers/apl_aro_adjust.py
