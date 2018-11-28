 

# Train source model
python adda.py -sds="para" 


# Train source model, no test evaluation
python adda.py -sds="para" -ting=0

# Train source model from weights
python adda.py -sds="para" -s="results/source_weights_para.h5"

# Eval source model on source para
python adda.py -t=True -s="results/source_weights_para.h5" -sds "para"
# Eval source model on target (acre)
python adda.py -t=True -s="results/source_weights_para.h5" -sds "acre"


# Domain training (No source classif. training involed)
python adda.py -f -s="results/source_weights_para.h5" -sds "para" -tds="acre"
# =========== Acre as source =======================================


# Train source model 
python adda.py -sds="acre" 

# Eval source model on source 
python adda.py -t=True -s="results/source_weights_acre.h5" -sds "acre"
# Eval source model on target
python adda.py -t=True -s="results/source_weights_acre.h5" -sds "para"

# Adversarial training
# First 2 epochs very good.... later not so much
python adda.py -f -s="results/source_weights_para.h5" -sds "para" -tds="acre"
# Now test it without any training: