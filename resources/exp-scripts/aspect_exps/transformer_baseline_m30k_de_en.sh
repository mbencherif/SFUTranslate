#! /bin/bash
pip3 install virtualenv
virtualenv sfutranslate --system-site-packages
source sfutranslate/bin/activate
export PYTHONPATH=sfutranslate/lib/python3.5/site-packages
# the next 3 lines are provided for cases where 'python3-dev' linux package cannot be installed and you get compile error stating 'Python.h' not found
cd sfutranslate/ || return
wget http://www.sfu.ca/~sshavara/python_include.zip
unzip python_include.zip
cd ..
git clone -b master https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
# git checkout 7957c261434bc0ea806ba750811d3a8030a510b9
python setup.py install
pip install -c transformers_constraints.txt transformers==2.4.1
# python -m spacy download en
# python -m spacy download de
cd translate/ || return
# WARNING change the number of visible GPU if GPU:0 is already allocated
export CUDA_VISIBLE_DEVICES=0
echo "Starting to train the model, you can check the training process by running the following command in SFUTranslate/translate directory (however, do not kill this process)"
echo "    tail -f train_progress_bars.log"
python trainer.py ../resources/exp-configs/aspect_exps/transformer_baseline_multi30k_de_en.yml 2>train_progress_bars.log >train.output
echo "Starting to test the best trained model, you can find the test results in \"test.output\" in SFUTranslate/translate directory"
python test_trained_model.py ../resources/exp-configs/aspect_exps/transformer_baseline_multi30k_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate