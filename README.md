# nmt-legislation-summarization
Using a seq2seq TensorFlow model to label sections of Canadian legislation

# Introduction

Sequence to sequence learning has been used to summarize text, by training with newspaper articles and their headlines [See this post from Google](https://ai.googleblog.com/2016/08/text-summarization-with-tensorflow.html) 

# Data

Another data set that contains text with corresponding labels is the body of [Canadian Legislation](http://laws.justice.gc.ca/eng/acts/) and [Regulations](http://laws.justice.gc.ca/eng/regulations/)

These documents contain "marginal notes" that describe each paragraph. [For example](http://laws.justice.gc.ca/eng/acts/A-1/page-1.html#h-6):


__Right to access to records__

*4 (1) Subject to this Act, but notwithstanding any other Act of Parliament, every person who is*

*(a) a Canadian citizen, or*

*(b) a permanent resident within the meaning of subsection 2(1) of the Immigration and Refugee Protection Act,*

*has a right to and shall, on request, be given access to any record under the control of a government institution.*

__Extension of right by order__

*(2) The Governor in Council may, by order, extend the right to be given access to records under subsection (1) to include persons not referred to in that subsection and may set such conditions as the Governor in Council deems appropriate.*

__Responsibility of government institutions__

*(2.1) The head of a government institution shall, without regard to the identity of a person making a request for access to a record under the control of the institution, make every reasonable effort to assist the person in connection with the request, respond to the request accurately and completely and, subject to the regulations, provide timely access to the record in the format requested.*

__Records produced from machine readable records__

*(3) For the purposes of this Act, any record requested under this Act that does not exist but can, subject to such limitations as may be prescribed by regulation, be produced from a machine readable record under the control of a government institution using computer hardware and software and technical expertise normally used by the government institution shall be deemed to be a record under the control of the government institution.*

The bold text provides a brief description of each paragraph that follows it.

Taken together, the Acts and Regulations have 113,611 labeled paragraphs, covering a range of topics. This dataset can be used to train a system that provides a label or short summary based on paragraph text. 

# Results

Here is an example of the results on ten randomly selected paragraphs from the test set. The predicted labels are given in bold with the actual labels following in parentheses:

1. the minister shall cause a notice of removal to be served on the person in respect of whom a notation is removed pursuant to subsection .  __duty to notify__ (duty to notify)

2. the principal purpose of animal pedigree associations shall be the registration and identification of animals and the keeping of animal pedigrees . __principal purpose__ (principal purpose)

3. no person holding shares or membership shares in the capacity of a personal representative and registered on the records of the bank as a shareholder or member and described in those records as the personal representative of a named person is personally liable under subsection but the named person is subject to all the liabilities imposed by that subsection . __shares__ (shares and membership shares held by personal representative)

4. for the purposes of subsection the total value of all assets that the bank or any of its subsidiaries has acquired during the period of twelve months referred to in subsection is the purchase price of the assets or if the assets are shares of or ownership interests in an entity the assets of which immediately after the acquisition were included in the annual statement of the bank the fair market value of the assets of the entity at the date of the acquisition . __total value of all assets__ (total value of all assets)

5. if the minister determines under subsection that a person committed a violation the person is liable to pay the amount of the penalty confirmed or corrected in that decision in the prescribed time and manner . __payment__ (payment)

6. this part does not apply to an award of damages to any of the following plaintiffs __non application of part__ (non application of part)

7. no person may give or accept in connection with the allocation or use of a part of a housing unit of the cooperative compensation that exceeds the amount that having regard to the portion of the housing unit would be a reasonable share of the housing charges for the housing unit determined in accordance with the by laws . __limit__ (limit on compensation)

8. the members shall appoint the first directors under paragraph c at a meeting held as soon as possible after the eight further members are appointed under subsection . __first directors__ (appointment of first directors under paragraph c)

9. if the minister makes a decision referred to in paragraph a the minister shall issue the directions under subsection that the minister considers appropriate and an employee may continue to refuse to use or operate the machine or thing work in that place or perform that activity until the directions are complied with or until they are varied or rescinded under this part . __directions under section__ (directions by minister)

10. subject to subsection no person to whom a detention order is addressed in accordance with subsection shall after receipt of the order give clearance in respect of the ship to which the order relates . __duty of persons empowered to give clearance__ (duty of persons authorized to give clearance)

The reported BLEU score by TensorFlow is 17.88

# Details

The model uses the TensorFlow [Neural Machine Translation (seq2seq) code](https://github.com/tensorflow/nmt/tree/tf-1.4). This model needs eight files: Source and target files for train, test, and dev sets, and a vocabulary file for the source and targets. 

The data files were built as follows:

`web_scrape.py:` saves all Acts/Regs locally

`extract_html_data.py:` uses BeautifulSoup python library to extract paragraphs and labels, and create a large text file. All labels are in h6 making them easy to extract. The paragraph following each label is the text:

``` python
small_titles = soup.find_all('h6')
        titl = []
        summary = []
        for ts in small_titles:
             if ts.next_sibling is not None:
                 if ts.contents[len(ts.contents)-1].string is not None:
                     titl.append(ts.contents[len(ts.contents)-1].string)
                     summary.append(' '.join(ts.next_sibling.find_all(text=True)))
```

`build_training_files.py:` Strip away punctuation (except periods), and sort randomly into train, test, and dev set, taking the first 50 words in each label and the first 200 in each paragraph. Build a vocabulary for the input and output (these could be shared, I have not examined the difference). The vocabulary consists of all words appearing more than once in each set. Others are treated as unknown. There are 14719 tokens making up the input vocabulary and 6522 making up the output vocabulary (so the model does cheat a bit by constraining the output to what we know is going to be in it). 
                 
# Running it

The way I trained it was on an AWS p2.xlarge instance, running the Deep Learning (Ubuntu) AMI. To run the tutorial code requires the TensorFlow nighlty distribution, which can be executed in a virtual environment as follows:

```
virtualenv --system-site-packages -p python3 ~/tensorflow
source ~/tensorflow/bin/activate 
easy_install -U pip
pip3 install tf-nightly-gpu 
```

To get set up:

```
mkdir acts_tf
cd acts_tf
mkdir html # first run, to store the html files pulled down from the internet
mkdir data
python web_scrape.py
python extract_html_data.py
python build_training_files.py
```
Optionally, re-run web_scrape and extract_html_data for the regulations, concatenate the output files with the ones resulting from the first run, and then build the training files.

Create a temp directory for the run and copy the data over:

```
cd ~/
mkdir /tmp/data
cp acts_tf/data/*.* /tmp/data
mkdir /tmp/nmt_model_acts
```

And train the model (about 5 hrs):

```
python -m nmt.nmt \
    --src=in --tgt=out \
    --vocab_prefix=/tmp/data/vocab  \
    --train_prefix=/tmp/data/train \
    --dev_prefix=/tmp/data/test  \
    --test_prefix=/tmp/data/dev \
    --out_dir=/tmp/nmt_model_acts \
    --num_train_steps=40000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
````

Run tensorboard to monitor:
```
tensorboard --port 8888 --logdir /tmp/nmt_model_acts/
```

Put random lines into inference_text.in (you could use compare_io.py to get some lines), then:

```
python -m nmt.nmt \
    --out_dir=/tmp/nmt_model_acts \
    --inference_input_file=acts_tf/inference_text.in \
    --inference_output_file=acts_tf/inference_result.txt
```


