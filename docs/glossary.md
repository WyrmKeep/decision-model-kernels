# Glossary

This glossary gives the terms of the project. Each term has one meaning. The checker reads this file.

## Columns

The table has three columns:

- Term: the word or the group of words of the row.
- Type: `technical noun`, `technical verb`, or `technical name`.
- Meaning: the information that the term gives to a person.

A term can be the same as an approved word of the STE dictionary. Then the project meaning is part of the approved meaning. Some meanings tell you which words you must not use. The glossary shows these words as code.

## Terms

| Term | Type | Meaning |
|---|---|---|
| American spelling | technical noun | The spelling that STE uses. Write `color`, not `colour`. |
| answer | technical noun | The output of the model for one question. |
| ASD | technical name | The group that makes the STE specification. |
| attention | technical noun | The step of a model layer that compares each token with the previous tokens. Do not use `self-attention`. |
| attention layer | technical noun | A model layer that does attention. It keeps data of the previous tokens in the KV cache. |
| bandwidth | technical noun | The number of bytes that a memory moves in one second. |
| baseline | technical noun | The build and the numbers that you compare with. Do not change a baseline after a test starts. |
| batch | technical noun | A group of tokens that the program gives to the model together. The flag `-b` sets its largest number of tokens. |
| benchmark | technical noun | A set of cases that does not change. It measures the speed and the quality of a server. |
| bit | technical noun | The smallest unit of data. It has the value 0 or 1. |
| branch | technical noun | A sequence of commits in a git repository. It has a name. |
| build | technical noun | The program that the compiler makes from one commit. |
| build | technical verb | To make a build. |
| cache | technical noun | A small memory that gives fast reads of data that you use many times. |
| case | technical noun | One item of the benchmark. A case has one state and one to three questions. |
| cell | technical noun | One item in a row of a table. |
| check | technical verb | To compare a text with the rules of STE. |
| checker | technical noun | The program `tools/ste/ste_check.py`. It finds text that does not follow the STE rules. |
| client | technical noun | A program that sends requests to a server. |
| clock | technical noun | The speed of one part of the GPU, in MHz. |
| column | technical noun | One vertical part of a table. |
| command | technical noun | The text that you write in a terminal to start a program. |
| commit | technical noun | One change that git keeps in a repository. A commit has a hash and a commit message. |
| commit | technical verb | To tell git to keep a change in a repository. |
| commit message | technical noun | The text that a person writes with a commit. It gives information about the change. |
| compiler | technical noun | A program that changes source code to a program that a device can run. |
| configuration file | technical noun | The file `tools/ste/config.toml`. It holds the limits and the lists of words of the checker. |
| contraction | technical noun | Two words that a person writes as one word with a mark, for example `don't`. |
| copyright | technical noun | The control that a person or a group has on the copies of a work. |
| CUDA | technical name | The system of NVIDIA that lets a program use a GPU. |
| CUDA graph | technical noun | A group of kernel launches. The program starts all of them with one launch. |
| decision | technical noun | One question with the answer of the model. |
| decode | technical noun | The part of a request in which the model writes the new tokens. It writes one token in each pass. |
| default | technical noun | The value that a program uses when you do not give a value. |
| description | technical noun | Text that gives information about an item. |
| dictionary | technical noun | The set of the approved words of STE. |
| divergence | technical noun | The condition when the threads of one warp do different steps of the code. |
| download | technical verb | To get a file from a server. |
| environment variable | technical noun | A value with a name that the terminal gives to a program. |
| exit code | technical noun | The number that a program gives when it stops. The number 0 shows that the program found no error. |
| false | technical name | The value of a condition that is not correct. |
| fault | technical noun | An error of a device or of a program that a log shows. |
| field | technical noun | One value with a name in a data file. |
| file | technical noun | A group of data with a name that a device keeps. |
| finding | technical noun | One line of the output of the checker. It shows one problem in the text. |
| flag | technical noun | A word that starts with `-` in a command. It changes the work of the program. |
| folder | technical noun | A group of files and other folders that has a name. |
| fork | technical noun | A copy of a repository that you can change. |
| fusion | technical noun | The change of two or more kernels to one kernel. It removes the steps that write and read data between them. |
| Gated DeltaNet | technical name | The type of recurrent layer in this model. |
| GGUF | technical name | The type of file that holds a model for llama.cpp. |
| git | technical name | The program that keeps all changes of the code in a repository. |
| glossary | technical noun | This file. It gives one meaning to each term. |
| GPU | technical name | The device that does the work of the kernels. In this project it is the `RTX 3060 Ti`. |
| graph | technical noun | The sequence of the operations that the model does for one batch. The program makes it again for each batch. |
| guard | technical noun | A limit on the quality of a run or on the memory that a run uses. A run must stay in all guards. |
| hash | technical noun | A short code that identifies one commit. |
| heading | technical noun | A line of text that gives a name to the text below it. |
| host | technical noun | A device that runs a server. |
| imperative verb | technical noun | A verb that tells a person to do a task. It is the first word of a step. |
| int8 | technical noun | A number that has 8 bits. Its value is from -128 to 127. |
| JSON | technical name | A type of text file for data with fields and lists. |
| kernel | technical noun | A program that runs on the GPU. It does one task on many numbers at the same time. |
| key | technical noun | The name of a field. |
| KV cache | technical noun | The cache of an attention layer. It keeps data of the previous tokens. |
| label | technical noun | The correct answer to one question of a case. A person writes it. |
| latency | technical noun | The time from the start of a request to the end of its response. |
| launch | technical noun | One start of a kernel on the GPU. Each launch uses a short time. |
| launch | technical verb | To tell the GPU to start a kernel. |
| layer state | technical noun | The numbers that a recurrent layer keeps between tokens. |
| letter | technical noun | One symbol of a word. |
| license | technical noun | The text that tells you when you can use the files. |
| line | technical noun | One row of text in a file. |
| list | technical noun | A sequence of items. |
| llama-server | technical name | The server program of llama.cpp. It receives the requests of a client and sends back the responses. |
| llama.cpp | technical name | The program that runs the model on a GPU. This project changes its code. |
| log | technical noun | A file that records the work of a program. |
| logit | technical noun | The number that the model gives to each possible next token before softmax. Softmax changes the logits to probabilities. |
| Markdown | technical name | A type of text file with marks for headings, lists, and tables. |
| matrix | technical noun | A table of numbers. |
| mean | technical noun | The sum of the values divided by their number. |
| meaning | technical noun | The information that a term gives to a person. |
| median | technical noun | The value in the middle when you put the values in sequence. |
| memory | technical noun | The part of a device that keeps data. |
| metric | technical noun | A number that measures one quality of a run. |
| model | technical noun | The model weights and the model layers that change tokens into logits. |
| model layer | technical noun | One step of the model that changes the data of each token. |
| model weight | technical noun | A number that the model keeps. It is part of a model layer. |
| notebook | technical noun | A file in the project that records each test with its command and its result. |
| noun | technical noun | A word for a person or an item. |
| noun cluster | technical noun | A group of nouns in sequence that name one item. |
| NVIDIA | technical name | The name of the group that makes the GPU and CUDA. |
| occupancy | technical noun | The part of the largest number of warps that are active on one SM. |
| paragraph | technical noun | A group of sentences with an empty line before it and after it. |
| pass | technical noun | One run of the model on one ubatch, from tokens to logits. Do not use `forward`. |
| passive verb | technical noun | A verb group, for example `is removed`. It does not show who does the work. |
| path | technical noun | The name of a file or a folder, with the names of the folders that hold it. |
| PDF | technical name | A type of file for text. The STE specification is a PDF. |
| power | technical noun | The electrical supply that a device uses, in watts. |
| power limit | technical noun | The largest power that the GPU is permitted to use. This project does not change it. |
| prefill | technical noun | The part of a request in which the model reads the prompt. It reads many tokens in one pass. |
| probability | technical noun | A number from 0 to 1 that shows how sure the model is of an answer. |
| profile | technical noun | The data about the time that each kernel uses in one run. |
| profile | technical verb | To record the time that each kernel uses in one run. |
| profiler | technical noun | A tool that makes a profile. |
| program | technical noun | A set of steps in code that a device can run. |
| project | technical noun | The work in this repository. |
| prompt | technical noun | The tokens that go into the model at the start of a request. |
| Python | technical name | The program that runs the script of the checker. |
| quantization | technical noun | The change of the model weights to a small number of bits for each weight. |
| quantize | technical verb | To change the model weights to a small number of bits for each weight. |
| question | technical noun | One item that the model must answer about a case. A question has a name. |
| recurrent layer | technical noun | A model layer that keeps a layer state between tokens. |
| reference | technical noun | A set of answers that you compare with the answers of a new build. |
| register | technical noun | A fast part of the memory of one thread. |
| repository | technical noun | A folder with the files of the project. Git keeps all changes of the files in it. |
| request | technical noun | The data that a client sends to the server for one case. |
| response | technical noun | The data that the server sends back for one request. |
| root folder | technical noun | The folder that has all other files of the repository. |
| rule | technical noun | One instruction of STE for the text. |
| run | technical noun | One time that a program does its work from start to stop. |
| run | technical verb | To start a program and let it work until it stops. |
| sampler | technical noun | A program that reads the memory use and the power of the GPU each second and writes the values to a file. |
| script | technical noun | A short program that a person writes in a file of text. |
| seed | technical noun | The start number of a random sequence. |
| semicolon | technical noun | The mark `;` in a sentence. |
| sentence | technical noun | A group of words that ends with a period. |
| server | technical noun | A program that waits for requests and sends back responses. |
| shared memory | technical noun | A fast memory on an SM. The threads of one thread block use it together. |
| SM | technical noun | One part of the GPU. It runs the threads of a thread block. The `RTX 3060 Ti` has 38 SMs. |
| softmax | technical noun | A function that changes logits to probabilities. The sum of the probabilities is 1. |
| specification | technical noun | The PDF that gives the rules and the dictionary of STE. |
| spelling | technical noun | The letters of a word in sequence. |
| state | technical noun | The text of one case. The model reads it before it answers the questions. |
| STE | technical name | The short name of `Simplified Technical English`. ASD-STE100 gives its rules. |
| string | technical noun | A sequence of text in a program or in a data file. |
| table | technical noun | Data in rows and columns. |
| tensor | technical noun | A group of numbers with a shape. |
| term | technical noun | A word or a group of words that has one meaning in the project. |
| terminal | technical noun | The program in which you write commands. |
| text | technical noun | Words and marks that a person can read. |
| thread | technical noun | The smallest unit of work in a kernel. A warp has 32 threads. |
| thread block | technical noun | A group of threads that use the same SM at the same time. |
| throughput | technical noun | The number of decisions that the server finishes in one second. |
| tile | technical noun | A small part of a matrix that one thread block reads and writes at one time. |
| token | technical noun | One piece of text that the model reads or writes. |
| tokenizer | technical noun | The program that changes text to tokens. |
| true | technical name | The value of a condition that is correct. |
| ubatch | technical noun | A part of a batch. The model reads one ubatch in each pass. The flag `-ub` sets its largest number of tokens. |
| verb | technical noun | A word for a task that a person or a device does. |
| VRAM | technical noun | The memory on the GPU. |
| warning | technical noun | A finding that is not an error. It does not change the exit code. |
| warp | technical noun | A group of 32 threads that do the same step at the same time. |
| word | technical noun | One unit of text between two spaces. |
