# miniature-winner

First, we needs the to install all pip requirements of NVIDIA's OpenSeq2Seq projects. The requirement file is located [here](https://github.com/USC-NSL/OpenSeq2Seq/blob/f4f56b800f53eceb4b4c30b33e20103da2a65432/requirements.txt) and [here](https://github.com/USC-NSL/miniature-winner/blob/master/requirements.txt)

    pip install -U -r requirements.txt
Then, we need to run tf-serving at port 8500. All current module will send RPC call to tf-serving at port 8500. I usually run my tf-serving using this [script](https://github.com/USC-NSL/miniature-winner/blob/master/script/run_tf_server.sh). 

The compressed file including all serverable can be downloaded [
here](https://drive.google.com/file/d/18RjXyF73ozk1ZHC32NMP-MECLG78J8TU/view?usp=sharing). It includes 5 models(3 for Speech Recognition[Deepspeech, Jasper, Wave2Letter], 1 for Translation[Transformer], and 1 for Speech Synthesis[Tacotron] ).

To run the pipeline, simply do

    python pipeline.py
in the root folder.


