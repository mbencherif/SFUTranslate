#! /bin/bash
pip3 install virtualenv
virtualenv sfutranslate
source sfutranslate/bin/activate
export PYTHONPATH=sfutranslate/lib/python3.5/site-packages
git clone -b dev-lingemb https://github.com/sfu-natlang/SFUTranslate.git
cd SFUTranslate/ || return
# git checkout 7957c261434bc0ea806ba750811d3a8030a510b9
python setup.py install
pip install -c transformers_constraints.txt transformers==2.4.1
# python -m spacy download en_core_web_lg
python -m spacy download de_core_news_md
cd translate/ || return
export PYTHONPATH=${PYTHONPATH}:`pwd`
cd models/aspect_extractor || return
python aspect_extract_main.py ../../../resources/exp-configs/aspect_exps/transformer_aspect_augmented_wmt19_de_en.yml 2>train_aspect_extractor.log >train_aspect_extractor.output
cd ../../ || return
echo "Starting to train the model, you can check the training process by running the following command in SFUTranslate/translate directory (however, fo not kill this process)"
echo "    tail -f train_progress_bars.log"
python trainer.py ../resources/exp-configs/aspect_exps/transformer_aspect_augmented_wmt19_de_en.yml 2>train_progress_bars.log >train.output
echo "Starting to test the best trained model, you can find the test results in \"test.output\" in SFUTranslate/translate directory"
python test_trained_model.py ../resources/exp-configs/aspect_exps/transformer_aspect_augmented_wmt19_de_en.yml 2>testerr.log >test.output
cat test.output # The dev and test scores are printed after this line
deactivate