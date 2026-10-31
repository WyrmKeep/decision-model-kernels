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
| ablation | technical noun | A test with a build that does not do the work of one operation class. The time that the test removes is the maximum gain for the operation class. |
| ablation build | technical noun | The build of the branch `dmk/ablate-timing`. It skips the work of the parts that an environment variable names. |
| access key | technical noun | A text that a program uses to get access to a server. |
| access token | technical noun | A text that a program gives to a server to get access. |
| accumulator | technical noun | The matrix of an MMA instruction that holds the sum. The instruction adds each new result to it. |
| activation | technical noun | The output of a model layer for one token. The next model layer reads it. |
| admin account | technical noun | A name on a server that can change all data and all adjustments of the server. |
| algorithm | technical noun | One kernel of cuBLASLt with its adjustments. cuBLASLt selects an algorithm for each GEMM. |
| aligned run | technical noun | A run of Nsight Compute in which the profiled launches are the launches of the list that the run must profile. |
| allow-list | technical noun | A list of paths. The file check lets each path in the list through. |
| AMD | technical name | The name of a group that makes a GPU. It is not NVIDIA. |
| American spelling | technical noun | The spelling that STE uses. Write `color`, not `colour`. |
| Ampere | technical name | A type of NVIDIA GPU. It comes after Turing. The `RTX 3060 Ti` has an Ampere GPU. |
| answer | technical noun | The output of the model for one question. |
| API | technical noun | The group of requests that a server accepts. The server sends back a response for each request. |
| Apple | technical name | The name of a group that makes devices. Each device has a CPU and a GPU that the group makes. |
| architecture | technical noun | The type of a GPU. The type sets the code that a kernel must contain for the GPU. The architecture of the `RTX 3060 Ti` is `sm_86`. |
| archive | technical noun | A file that holds many files in one file, for example a `.tar` file. |
| arena | technical noun | A memory pool that only one part of the program uses. |
| arm | technical noun | One set of programs that a test compares with a different set. An arm is `stock` or `build:<name>`. |
| ASD | technical name | The group that makes the STE specification. |
| assumption | technical noun | A fact or a number that a prediction uses and that we did not measure. |
| attention | technical noun | The step of a model layer that compares each token with the previous tokens. Do not use `self-attention`. |
| attention layer | technical noun | A model layer that does attention. It keeps data of the previous tokens in the KV cache. |
| attention mask | technical noun | A matrix with one value for each pair of a token and a KV cell. The value shows if the token can use the data of that KV cell. |
| author | technical noun | The person who wrote an issue or a pull request. |
| autoregressive kernel | technical noun | A GDN kernel that goes through the tokens of one sequence one at a time. |
| backend | technical noun | A part of llama.cpp that runs the graph on one type of device. CUDA is one backend. |
| backend sampling | technical noun | A mode in which the backend selects the next token. The host then does not read all logits. The flag `-bs` starts it. |
| bandwidth | technical noun | The number of bytes that a memory moves in one second. |
| barrier | technical noun | An instruction in a kernel. A thread waits at it until all threads of its thread block come to it. |
| base file | technical noun | The file that `llama perplexity` writes with the flag `--kl-divergence-base`. It keeps the log-probabilities of the tokens of a text for one build. |
| baseline | technical noun | The build and the numbers that you compare with. Do not change a baseline after a test starts. |
| batch | technical noun | A group of tokens that the program gives to the model together. The flag `-b` sets its largest number of tokens. |
| benchmark | technical noun | A set of cases that does not change. It measures the speed and the quality of a server. |
| bias | technical noun | A model weight that a model layer adds to its output. |
| binary file | technical noun | A file that holds bytes that a person cannot read as text. |
| bit | technical noun | The smallest unit of data. It has the value 0 or 1. |
| bit mask | technical noun | A number in which each bit has a different meaning. Each bit shows if one condition is true. |
| Blackwell | technical name | A type of NVIDIA GPU. It comes after Hopper. |
| blobless clone | technical noun | A copy of a repository that has all commits but not the data of the files. Git downloads a file when a command must read it. |
| block | technical noun | A group of runs in a schedule. It has one run of each arm, in a random sequence. |
| bootstrap | technical noun | A method that makes many new sets of values from one set. It selects values from the set at random. It can select the same value more than one time. |
| bot | technical noun | A program that writes comments or closes items on GitHub without a person. |
| branch | technical noun | A sequence of commits in a git repository. It has a name. |
| budget | technical noun | The quantity of work that the owner gives to the project. |
| buffer | technical noun | A part of the memory that a program uses for data. |
| build | technical noun | The program that the compiler makes from one commit. |
| build | technical verb | To make a build. |
| builder image | technical noun | The image that has the compiler and the other programs that build llama.cpp. The script `build.sh` builds in a container from it. |
| cache | technical noun | A small memory that gives fast reads of data that you use many times. |
| cache key | technical noun | The value that a program uses to find an entry in a cache. |
| cap | technical noun | The time of the arm `base` minus the time of an arm of an ablation. It is the largest time that a faster part can remove from a request. |
| capability | technical noun | One type of task that Linux lets a program do. The flag `--cap-add` of Docker gives a container one capability. |
| capture | technical verb | To record the kernel launches of one pass in a CUDA graph. The program can then start all of them with one launch. |
| capture | technical noun | The file that Nsight Systems writes for one run of the server. It has the times of the kernels, the copies, and the CUDA calls. |
| carrier-grade NAT range | technical noun | A range of IP addresses that RFC 6598 gives for internal use in large networks. |
| case | technical noun | One item of the benchmark. A case has one state and one to three questions. |
| CDNA | technical name | A type of AMD GPU. It has the instruction `MFMA`, which multiplies two tiles. |
| ceiling | technical noun | The largest speed increase that we think a change can give. We calculate it before a test. |
| cell | technical noun | One item in a row of a table. |
| census | technical noun | The work of each pass in numbers. The numbers come from the shapes of the model and the rules of the code. They are not time measurements. |
| chain | technical noun | A sequence of MMA instructions in one warp. Each instruction uses the result of the instruction before it. |
| character | technical noun | One symbol of a text, for example a letter, a number, or a space. |
| check | technical verb | To compare a text with the rules of STE. |
| checker | technical noun | The program `tools/ste/ste_check.py`. It finds text that does not follow the STE rules. |
| checkpoint | technical noun | A copy of the data of a slot at one token position. The server can go back to that position with it. |
| checksum | technical noun | A short code that a program calculates from the data of a file. The code changes when the data changes. |
| cherry-pick | technical verb | To put one commit of a branch onto a different branch with git. |
| child slot | technical noun | The task slot of a child task. |
| child task | technical noun | A task of a group of questions that is not the parent task. It waits for the parent task. The server then copies the data of the parent task to it. |
| chunk | technical noun | A part of a sequence of tokens or of a graph. A program uses each chunk as one unit. |
| chunked kernel | technical noun | A GDN kernel that divides the tokens of one sequence into chunks. It is for prefill. |
| CI | technical name | The tests that a server runs after each git push. The server starts them automatically. |
| client | technical noun | A program that sends requests to a server. |
| clock | technical noun | The speed of one part of the GPU, in MHz. |
| clone | technical noun | A copy of a repository on one host. It has the history of the repository. |
| CMake | technical name | The program that makes the build files of llama.cpp and starts the compiler. |
| column | technical noun | One vertical part of a table. |
| command | technical noun | The text that you write in a terminal to start a program. |
| command buffer | technical noun | The memory of the CUDA driver that holds the launches before the GPU starts them. |
| comment | technical noun | A text that a person writes below an issue or a pull request on GitHub. |
| commit | technical noun | One change that git keeps in a repository. A commit has a hash and a commit message. |
| commit | technical verb | To tell git to keep a change in a repository. |
| commit message | technical noun | The text that a person writes with a commit. It gives information about the change. |
| compiler | technical noun | A program that changes source code to a program that a device can run. |
| compute capability | technical noun | A number of two parts for the type of an NVIDIA GPU, for example 8.6. The name `sm_86` has the same number. |
| concat | technical noun | The operation that puts two tensors together in one tensor along one dimension. |
| confidence | technical noun | The probability that a change passes the test of its equivalence class and gives 50 % or more of the gain of its prediction. We give it before the test. |
| confidence interval | technical noun | A range of values that we calculate from the data of a test. The true value is in the range with a probability that we write with the range, for example 95 %. |
| configuration file | technical noun | The file `tools/ste/config.toml`. It holds the limits and the lists of words of the checker. |
| contraction | technical noun | Two words that a person writes as one word with a mark, for example `don't`. |
| conv state | technical noun | A part of the layer state of a Gated DeltaNet layer. It has the last inputs of the layer. The kernel `ssm_conv` reads it. |
| convolution | technical noun | The step of a recurrent layer that mixes each token with the tokens before it. The node `SSM_CONV` does it. |
| copyright | technical noun | The control that a person or a group has on the copies of a work. |
| correctness | technical noun | The quality of a build that gives the correct answers. |
| CPU | technical name | The part of a host that does the work of most programs. The GPU does the work of the kernels. |
| credential helper | technical noun | A program that gives a secret to a different program, for example to git or to a script. |
| CSV | technical name | A type of text file for data. In each line, the mark `,` is between two values. |
| cubin | technical noun | A file with the SASS of one part of a library for one architecture. The program `cuobjdump` writes it from the library. |
| cuBLASLt | technical name | The library of NVIDIA that multiplies one matrix with a different matrix on the GPU. It selects an algorithm for each GEMM. |
| CUDA | technical name | The system of NVIDIA that lets a program use a GPU. |
| CUDA call | technical noun | One use of a function of the CUDA runtime, for example `cudaLaunchKernel` or `cudaMemcpyAsync`. |
| CUDA graph | technical noun | A group of kernel launches. The program starts all of them with one launch. |
| CUDA runtime | technical name | The library of CUDA that a program uses to give kernels to the GPU. |
| CUDA stream | technical noun | A list of work for the GPU. The GPU does the work of one CUDA stream in sequence. It can do the work of different CUDA streams at the same time. |
| CUTLASS | technical name | The library of NVIDIA that has kernels for matrix operations on the GPU. |
| D | technical name | The short name of the build of the tag `b12004` that the project makes from source for `sm_86` only. Arm D uses it. |
| data set | technical noun | The data of one test. It is in one folder of the folder `data`. It has raw files, derived tables, and a README. |
| day | technical noun | A unit of time. One day has 24 h. |
| decimal place | technical noun | One position after the point in a number. The number 3.25 has two decimal places. |
| decision | technical noun | One question with the answer of the model. |
| declare | technical verb | To write a value in a document before a test. We do not change the value after the test starts. |
| decode | technical noun | The part of a request in which the model writes the new tokens. It writes one token in each pass. |
| decomposition | technical noun | The wall time of a request in steps. The sum of the steps is the wall time. |
| default | technical noun | The value that a program uses when you do not give a value. |
| dense model | technical noun | A model that has no MoE layer. |
| dense rate | technical noun | The rate of a tensor core for a matrix with the usual values. It is not the rate for a matrix with many zero values. |
| derived table | technical noun | A table that a script makes from raw files. |
| description | technical noun | Text that gives information about an item. |
| dictionary | technical noun | The set of the approved words of STE. |
| digest | technical noun | The SHA-256 checksum of an image. It names one version of the image. It starts with `sha256:`. |
| discussion | technical noun | A group of texts in a repository on GitHub. It is not an issue and it is not a pull request. It has a number. |
| disk | technical noun | A device in a host that keeps files. The files stay on it when the host is off. |
| Docker | technical name | The program that starts a container from an image on a host. |
| document | technical noun | A file of text that gives information about the project. A README is an example. |
| domain name | technical noun | The name of a group of hosts in a network, for example `example.net`. |
| download | technical verb | To get a file from a server. |
| draft | technical noun | A small and fast part that writes the next tokens before the model checks them. |
| draft pull request | technical noun | A pull request that the person who wrote it marks as not complete. A person cannot merge it. |
| DRAM | technical name | The memory parts on the GPU that hold the VRAM. The GPU gets data from the L2 cache faster than from DRAM. |
| drift | technical noun | A slow change of the speed of the host during a test. |
| driver | technical noun | The program that lets the other programs use a device, for example the GPU. |
| effort | technical noun | The work to write and test a change. The values are `S` (1 day or less), `M` (2 to 5 days), `L` (1 to 2 weeks) and `XL` (more than 2 weeks). |
| element-wise kernel | technical noun | A kernel that does the same operation on each value of a tensor. A value does not use the values at other positions. |
| email address | technical noun | The text with the mark `@` that names one person on a network, for example `name@example.com`. |
| embedding | technical noun | The model weights that give a list of numbers for each token. |
| env file | technical noun | The file with the environment variables of the harness. The environment variable `DMK_ENV` gives its path. |
| environment variable | technical noun | A value with a name that the terminal gives to a program. |
| equal split | technical noun | The method that the program uses to divide a batch into ubatches. In each ubatch all sequences have the same number of tokens. The function `split_equal` does it. |
| equivalence class | technical noun | A group of changes that must have the same tests of correctness. The three groups are `exact`, `re-associated`, and `approximation`. |
| equivalence margin | technical noun | The largest difference between two builds that we accept as no difference. We declare it before a test. |
| exit code | technical noun | The number that a program gives when it stops. The number 0 shows that the program found no error. |
| expected gain | technical noun | The gain of a prediction multiplied by the confidence. It is not a measurement. |
| expert | technical noun | One of the feed-forward layers of an MoE layer. |
| false | technical name | The value of a condition that is not correct. |
| false positive | technical noun | An item that gitleaks finds in a text that has no private data. |
| fault | technical noun | An error of a device or of a program that a log shows. |
| feed-forward layer | technical noun | A model layer that changes the data of each token. It does not compare the tokens with each other. |
| field | technical noun | One value with a name in a data file. |
| file | technical noun | A group of data with a name that a device keeps. |
| file check | technical noun | The check of the publication gate for the number of bytes and the type of each file. |
| finding | technical noun | One line of the output of the checker. It shows one problem in the text. |
| fingerprint | technical noun | The text `commit:file:rule:line`. It names one item that gitleaks finds. |
| fix-up kernel | technical noun | The kernel that follows an MMQ kernel with stream-k. It adds the results of different thread blocks for the same tile. |
| fixed-order model | technical noun | A model of the passes of a request with more than one question. The parent task has the lowest task slot. All child tasks are in one batch. |
| flag | technical noun | A word that starts with `-` in a command. It changes the work of the program. |
| flash attention | technical noun | A method to do attention in one kernel. The kernel works on tiles and does not write the large matrix of the attention step to memory. |
| floating point | technical noun | A type of number with a part after the point. FP16 and FP32 are floating point types. |
| floor | technical noun | The minimum time that a pass or a request can use. A limit of the GPU sets it, for example the DRAM rate. |
| folder | technical noun | A group of files and other folders that has a name. |
| force push | technical noun | A git push that replaces the history of a ref in the remote. |
| fork | technical noun | A copy of a repository that you can change. |
| fuse | technical verb | To make a fusion of two or more kernels. |
| fusion | technical noun | The change of two or more kernels to one kernel. It removes the steps that write and read data between them. |
| gain | technical noun | The time that a change removes from the time of a request. A prediction or a measurement gives it, in ms or in %. |
| gap | technical noun | The idle time of the GPU before a pass or between two kernels. |
| garbage collection | technical noun | The work of git or Gitea that removes the data that no ref holds. |
| Gated DeltaNet | technical name | The type of recurrent layer in this model. |
| gather | technical noun | The operation that reads pieces of data from different positions and writes them together in one new area. |
| gather | technical verb | To read pieces of data from different positions and write them together in one new area. |
| GDN | technical name | The short name of Gated DeltaNet. |
| GEMM | technical name | The operation that multiplies one matrix with a different matrix. A model layer does it with its weights and the activations of the tokens. |
| GFLOP | technical name | The unit of 10^9 floating point operations. A multiply-add counts as 2 operations. |
| ggml | technical name | The library of llama.cpp that has the tensors, the graph, the scheduler, and the backends. |
| GGUF | technical name | The type of file that holds a model for llama.cpp. |
| git | technical name | The program that keeps all changes of the code in a repository. |
| git push | technical noun | One use of the command `git push`. It sends commits to a remote. |
| Gitea | technical name | The git server that the project uses for its work. Only the persons of the project can read it. |
| gitleaks | technical name | A program that reads the files and the history of a repository. It finds the text that a gitleaks rule marks. |
| gitleaks rule | technical noun | One entry in a `.toml` file for gitleaks. It marks one type of text. |
| global memory | technical noun | The memory of the GPU that all thread blocks can read and write. It is in the VRAM. |
| glossary | technical noun | This file. It gives one meaning to each term. |
| GLU | technical name | A step of a feed-forward layer. It multiplies two lists of numbers of the layer, number by number. |
| GPU | technical name | The device that does the work of the kernels. In this project it is the `RTX 3060 Ti`. |
| GPU busy time | technical noun | The time in which one kernel or more run on the GPU. |
| GPU span | technical noun | The time from the start of the first pass of a request to the end of the last wait for the GPU. It is the maximum of the GPU time. |
| GPU wait | technical noun | The time in which the decode thread waits for the GPU to complete the work of a pass. |
| graph | technical noun | The sequence of the operations that the model does for one batch. The program makes it again for each batch. |
| graph plan | technical noun | An object that a backend makes from a graph one time. The backend can use it for many runs of the graph. |
| grid | technical noun | All the thread blocks of one launch of a kernel. |
| guard | technical noun | A limit on the quality of a run or on the memory that a run uses. A run must stay in all guards. |
| H2O Lightning | technical name | The name of the model of this project. The model is a decision model. |
| harness | technical noun | The scripts in the folder harness. They run a test on a device that has a GPU and keep the data of the test. |
| hash | technical noun | A short code that identifies one commit. |
| head dimension | technical noun | The number of values that one equal part of an attention layer or of a recurrent layer keeps for each token. |
| heading | technical noun | A line of text that gives a name to the text below it. |
| Hexagon | technical name | A type of device that Qualcomm makes. llama.cpp has a backend for it. |
| HIP | technical name | The system of AMD that lets a program use a GPU of AMD. CUDA does the same work for NVIDIA. |
| history | technical noun | The sequence of all commits of a branch or a tag, from the first commit. |
| home server | technical noun | A small server that one person has for the work of that person. A group of persons does not use it. |
| hook | technical noun | A program that git starts before or after a step, for example before a git push. |
| Hopper | technical name | A type of NVIDIA GPU. It comes after Ampere. |
| host | technical noun | A device that runs a server. |
| HTTP | technical name | The type of text that a client and a server send to each other on a network. A request and a response are examples. |
| Hugging Face | technical name | A server that keeps model files. You can download the files from it. |
| hybrid memory | technical noun | The memory code of llama.cpp for a hybrid model. It keeps the KV cache of the attention layers and the layer states of the recurrent layers. |
| hybrid model | technical noun | A model that has attention layers and recurrent layers. |
| I | technical name | The short name of the build of `L` with the change 0007. Arm I uses it. |
| idle time | technical noun | The time in which the GPU runs no kernel. |
| image | technical noun | A program with all the files that it must have. A container tool starts a container from an image. |
| image ID | technical noun | The SHA-256 name that a host gives to a local image. |
| IMMA | technical name | The instruction of a tensor core that multiplies int8 tiles. It is the int8 case of the MMA instruction. |
| imperative verb | technical noun | A verb that tells a person to do a task. It is the first word of a step. |
| INFERENCE | technical name | The mark for a sentence with information that we did not measure. We know the information from code or from text only. |
| int8 | technical noun | A number that has 8 bits. Its value is from -128 to 127. |
| int8 peak | technical noun | The largest number of int8 operations in one second that the tensor cores can do. The scripts of the project use 149.3 TOP/s. |
| integer | technical noun | A number that does not have a part after the point, for example 3 or -7. |
| IP address | technical noun | The number that names a host in a network. |
| issue | technical noun | A report or a question in a repository on GitHub. It has a number. |
| JSON | technical name | A type of text file for data with fields and lists. |
| kernel | technical noun | A program that runs on the GPU. It does one task on many numbers at the same time. |
| kernel class | technical noun | A group of kernels that do the same task, for example `MMQ`, `MMVQ`, or `GDN`. |
| key | technical noun | The name of a field. |
| KL budget | technical noun | The KL divergence of the model mix8 from the Q8_0 model. A change to a kernel must add much less KL divergence than this value. |
| KL divergence | technical noun | A number that shows how much the probabilities of the next token from one build are different from the probabilities from a different build. The value 0 shows no difference. |
| KL floor | technical noun | The KL divergence of a run against a base file that the same build wrote. It is small and not 0. |
| KV cache | technical noun | The cache of an attention layer. It keeps data of the previous tokens. |
| KV cell | technical noun | One part of the KV cache that keeps the data of one token. |
| KV window | technical noun | The range of KV cells that the kernel for flash attention reads for one sequence. |
| L | technical name | The short name of the build of `D` with the patch of pull request #30489. Arm L uses it. |
| L2 cache | technical noun | The cache that all SMs of the GPU use together. It is between the SMs and the DRAM. |
| label | technical noun | The correct answer to one question of a case. A person writes it. |
| latency | technical noun | The time from the start of a request to the end of its response. |
| launch | technical noun | One start of a kernel on the GPU. Each launch uses a short time. |
| launch | technical verb | To tell the GPU to start a kernel. |
| launch queue | technical noun | The launches that the CPU sent and that the GPU did not start. |
| layer state | technical noun | The numbers that a recurrent layer keeps between tokens. |
| layout | technical noun | The sequence of the values of a matrix in memory, for example `col` or `col32`. |
| lead | technical noun | A change from the code of upstream that we measure to see its effect. The project does not use it in production. |
| leading dimension | technical noun | The number of values from the start of one group of columns of a matrix to the start of the next group. The code uses the name `ld`. |
| letter | technical noun | One symbol of a word. |
| library | technical noun | A set of programs that other programs can use. |
| license | technical noun | The text that tells you when you can use the files. |
| line | technical noun | One row of text in a file. |
| link | technical noun | A part of a document that sends a person to a different file. |
| Linux | technical name | The primary program of a host. It starts all other programs and gives them the memory and the devices. |
| list | technical noun | A sequence of items. |
| llama-server | technical name | The server program of llama.cpp. It receives the requests of a client and sends back the responses. |
| llama.cpp | technical name | The program that runs the model on a GPU. This project changes its code. |
| load | technical verb | To read the files of a model into the memory of a program. |
| load average | technical noun | The numbers that show how much work the CPU of a host has. The file `/proc/loadavg` gives them. |
| log | technical noun | A file that records the work of a program. |
| logit | technical noun | The number that the model gives to each possible next token before softmax. Softmax changes the logits to probabilities. |
| long request | technical noun | A request for a case with a long state. The state has about 2,900 to 3,400 tokens. |
| macro | technical noun | A name in source code that the compiler changes to other code. |
| maintainer | technical noun | A person who has the right to merge pull requests into llama.cpp. |
| manifest | technical noun | The file `MANIFEST.txt` that the script `build.sh` writes in `dist/`. It has the facts of one build. |
| mapped host memory | technical noun | The memory of the host that the GPU can read and write directly. |
| Markdown | technical name | A type of text file with marks for headings, lists, and tables. |
| matmul | technical noun | One GEMM of a weight matrix with the activations of the tokens of one pass. An MMQ kernel or an MMVQ kernel does it. |
| matrix | technical noun | A table of numbers. |
| mean | technical noun | The sum of the values divided by their number. |
| meaning | technical noun | The information that a term gives to a person. |
| measurement | technical noun | The data that one run of a test gives. It has a value and a unit. |
| measurement window | technical noun | The time in which the project does the runs of a test on the GPU. The GPU does the work of one run at a time. |
| median | technical noun | The value in the middle when you put the values in sequence. |
| medium request | technical noun | A request for a case with a state of moderate length. The state has about 1,000 to 1,300 tokens. |
| megakernel | technical noun | One kernel that does the work of many kernels, for example the work of all kernels of one model layer. |
| memory | technical noun | The part of a device that keeps data. |
| memory bus | technical noun | The connection between the GPU and the DRAM. The `RTX 3060 Ti` has a memory bus of 256 bit. |
| memory cell | technical noun | One unit of the memory of a model. It holds the data of one token in the KV cache, or the layer state of one sequence. |
| memory pool | technical noun | A block of VRAM that the CUDA backend keeps for the temporary data of a graph. The backend gives out parts of it. |
| merge | technical verb | To put the commits of a pull request into a branch. |
| merge commit | technical noun | A commit that puts the changes of two branches together. |
| merge conflict | technical noun | The condition when two branches have different changes for the same line. A person must select the correct change. |
| metric | technical noun | A number that measures one quality of a run. |
| mirror sync | technical noun | One time that a push mirror sends the refs of a repository to a different server. |
| misaligned run | technical noun | A run of Nsight Compute in which the profiled launches are not the launches of the list that the run must profile. |
| MMA | technical name | The instruction of an NVIDIA GPU that multiplies two tiles and adds the result to a third tile. A kernel that uses it is an MMA kernel. |
| MMQ | technical name | The CUDA kernels of llama.cpp that multiply quantized model weights with int8 numbers. |
| MMVQ | technical name | The CUDA kernels of llama.cpp that multiply quantized model weights with a batch that has a small number of columns. MMQ is for a batch that has more columns. |
| model | technical noun | The model weights and the model layers that change tokens into logits. |
| model layer | technical noun | One step of the model that changes the data of each token. |
| model weight | technical noun | A number that the model keeps. It is part of a model layer. |
| MoE | technical name | The type of model layer that has many feed-forward layers. The model uses only some of them for each token. |
| MTP | technical name | A type of draft. The model has more layers that write the next tokens. |
| multiply-add | technical noun | The operation that multiplies two numbers and adds the result to a third number. A test counts it as 2 operations. |
| MUSA | technical name | A system that lets a program use a GPU, as CUDA does. A MUSA GPU is not an NVIDIA GPU. |
| network | technical noun | The links between hosts. A host sends data to a different host through the network. |
| NMSE | technical name | A number that shows the difference between a result and a reference result. A value of 0 shows no difference. The tests of `test-backend-ops` use a limit for it. |
| node | technical noun | One operation of a graph. |
| nominal rate | technical noun | The highest rate that you can calculate from the architecture of the GPU. A test does not measure it. |
| non-causal attention | technical noun | A type of attention in which each token compares with the previous tokens and with the next tokens. |
| notebook | technical noun | A file in the project that records each test with its command and its result. |
| noun | technical noun | A word for a person or an item. |
| noun cluster | technical noun | A group of nouns in sequence that name one item. |
| Nsight Compute | technical name | The profiler of NVIDIA for one kernel. It shows how the kernel uses the GPU. |
| Nsight Systems | technical name | The profiler of NVIDIA for one run of a program. It shows when each kernel and each launch starts and stops. |
| NVIDIA | technical name | The name of the group that makes the GPU and CUDA. |
| NVML | technical name | The library of NVIDIA that gives the clock, the power and the temperature of the GPU to a program. `nvidia-smi` uses it. |
| NVRTC | technical name | The library of CUDA that has a compiler for kernels. A program can use it while the program runs. |
| occupancy | technical noun | The part of the largest number of warps that are active on one SM. |
| opcode | technical noun | The name of one SASS instruction, for example `IMMA`, `LDS`, or `FFMA`. |
| operand | technical noun | A matrix that an MMA instruction reads. |
| operation class | technical noun | A group of kernels that do the same type of operation, for example the MMQ kernels or the GDN kernels. |
| output head | technical noun | The last model weights of the model. They change the data of one token to one logit for each token of the vocabulary. |
| overhead | technical noun | The time that a program uses for steps that are not the task, for example a kernel launch. |
| override | technical noun | A value that a schedule line sets for that line only. It replaces the value of the env file for the run of that line. |
| owner | technical noun | The person who selects the next step of the project. |
| owner account | technical noun | The name that the project uses on GitHub and on Gitea for its repositories. |
| P-state | technical noun | A power mode of the GPU. The mode P0 is the fastest mode. The mode P8 is for no load. |
| padding | technical noun | The tokens that the program adds to a sequence or to a ubatch to make it longer. |
| paragraph | technical noun | A group of sentences with an empty line before it and after it. |
| parameter | technical noun | A value that a program gives to a function or to an operation. |
| parent slot | technical noun | The task slot of a parent task. |
| parent task | technical noun | The first task of a group of questions. It runs the model on the shared prefix of the group. |
| Pascal | technical name | A type of NVIDIA GPU. It comes before Volta. |
| pass | technical noun | One run of the model on one ubatch, from tokens to logits. Do not use `forward`. |
| pass spread | technical noun | The difference between the highest and the lowest mean latency of the timed passes of one run, in % of the mean of the pass means. |
| passive verb | technical noun | A verb group, for example `is removed`. It does not show who does the work. |
| path | technical noun | The name of a file or a folder, with the names of the folders that hold it. |
| PCI address | technical noun | The text that names the slot of a GPU in a host. |
| PCIe | technical name | The connection between the GPU and the host. The version and the width of the slot give the rate of the copies. |
| PDF | technical name | A type of file for text. The STE specification is a PDF. |
| percentile | technical noun | A value in a list of numbers. Some numbers of the list are below it. The 95th percentile has 95 % of the numbers below it. |
| perplexity | technical noun | A number that the program calculates from the probabilities that the model gives to the tokens of a text. A smaller number shows a better model for that text. |
| persistent kernel | technical noun | A kernel that stays on the GPU and gets new work with no new launch. |
| phase | technical noun | A part of a run of the client `bench.py`. A phase has a name, a start, and an end. The names are `warmup`, `quality`, `seq`, `tput`, and `cold`. |
| pipe | technical noun | A group of units of an SM that run one type of instruction. The IMMA pipe runs the IMMA instructions. |
| pipeline | technical noun | A kernel with steps that overlap. The kernel reads the next tile from memory while it uses the tile that it read before. |
| pooling | technical noun | The step that puts the output of all tokens of one sequence into one list of numbers. |
| port | technical noun | A number that a server uses on a host to receive requests. A client sends its requests to this number. |
| power | technical noun | The electrical supply that a device uses, in watts. |
| power cap | technical noun | The condition in which the driver lowers the clock of the GPU to keep the power below the power limit. |
| power limit | technical noun | The largest power that the GPU is permitted to use. This project does not change it. |
| predict | technical verb | To write a prediction. |
| prediction | technical noun | The result that we think a test will give. We write it before the test. |
| prefill | technical noun | The part of a request in which the model reads the prompt. It reads many tokens in one pass. |
| prefix | technical noun | The first tokens of a prompt. |
| private benchmark | technical noun | The benchmark that the project used before it had the public benchmark. Only the persons of the project can read it. |
| private data | technical noun | Data that only the persons of the project can read. Secrets, internal names, and server paths are private data. |
| private rules file | technical noun | The file with the gitleaks rules for the internal names of the project. Only the persons of the project can read it. |
| probability | technical noun | A number from 0 to 1 that shows how sure the model is of an answer. |
| production | technical noun | The condition of a build that the live decision server runs. The server answers the requests of clients. |
| profile | technical noun | The data about the time that each kernel uses in one run. |
| profile | technical verb | To record the time that each kernel uses in one run. |
| profiler | technical noun | A tool that makes a profile. |
| program | technical noun | A set of steps in code that a device can run. |
| programmatic dependent launch | technical noun | The function of CUDA that lets a kernel start before the kernel that comes before it stops. |
| project | technical noun | The work in this repository. |
| projection | technical noun | A matrix multiply of the activations with one weight matrix of a layer. A feed-forward layer has three projections: `ffn_up`, `ffn_gate` and `ffn_down`. |
| prompt | technical noun | The tokens that go into the model at the start of a request. |
| prompt cache | technical noun | A cache of the server. It keeps the data of slots for previous prompts. The server uses the data again when a new prompt has the same prefix. |
| proposal | technical noun | An issue or a pull request that tells the project to make a change that the code does not have. |
| public benchmark | technical noun | The benchmark in this repository. Each person can read it. |
| public repository | technical noun | A repository that each person can read. The public repository of the project is on GitHub. |
| publication gate | technical noun | The scripts in the folder `tools/publish`. They stop private data before it goes to the public repository. |
| publish | technical verb | To send the data of the repository in Gitea to the public repository on GitHub. |
| pull request | technical noun | A request on GitHub to put the changes of one branch into a different branch. It has a number. |
| push mirror | technical noun | A function of Gitea. It sends the refs of a repository to a different server, for example GitHub. |
| Python | technical name | The program that runs the script of the checker. |
| Qualcomm | technical name | The name of a group that makes devices. |
| quantization | technical noun | The change of the model weights to a small number of bits for each weight. |
| quantize | technical verb | To change the model weights to a small number of bits for each weight. |
| query | technical noun | The text that a client sends to a server to find items. The response gives the items that have the words of the text. |
| question | technical noun | One item that the model must answer about a case. A question has a name. |
| ragged batching | technical noun | A method to put sequences of different lengths in one batch. The program adds no tokens to make the lengths equal. |
| rank | technical noun | The position of a target in the ranked list. The first target has rank 1. |
| rank | technical verb | To write the targets in a list. The target with the largest possible time decrease is first. |
| ranked list | technical noun | The list of the targets in the sequence of their expected gain. The target with the largest expected gain is first. |
| ratio | technical noun | The value of one number divided by a different number. |
| raw file | technical noun | A file that a program wrote. A person does not change it after that. |
| reaction | technical noun | A small symbol that a person adds to an issue, a pull request, or a comment on GitHub. |
| README | technical name | The file README.md in a folder. It gives the first information about the folder. |
| rebase | technical noun | A change of the commit at the start of a branch. Git puts the commits of the branch on a different commit. The command `git rebase` does it. |
| recurrent layer | technical noun | A model layer that keeps a layer state between tokens. |
| ref | technical noun | A name in git that points to a commit. A branch and a tag are refs. |
| reference | technical noun | A set of answers that you compare with the answers of a new build. |
| register | technical noun | A fast part of the memory of one thread. |
| relative link | technical noun | A link that gives the path of a file from the folder of the document that has the link. |
| release | technical noun | A set of files of llama.cpp that a person can download from GitHub. Each release has a tag. |
| remote | technical noun | A repository on a different host. A clone sends commits to it and gets commits from it. |
| repack | technical verb | To write the model weights again in a new sequence in memory. A kernel then reads them faster. |
| repository | technical noun | A folder with the files of the project. Git keeps all changes of the files in it. |
| reproduce | technical verb | To do a test again with the same commit, the same flags, and the same data. |
| request | technical noun | The data that a client sends to the server for one case. |
| request-start gap | technical noun | The gap before the first pass of a request. |
| response | technical noun | The data that the server sends back for one request. |
| REST API | technical name | A type of API. Each request names one item, and the response gives the data of that item. |
| reviewer | technical noun | A person who reads the changes of a pull request and writes comments. |
| revision | technical noun | The name of a commit. It can be a hash, a branch, or a tag. |
| rewrite | technical noun | The change of the commits of a branch to new commits with new hashes. A rewrite changes the history. |
| rewrite | technical verb | To make new commits with new hashes for the commits of a branch. This changes the history. |
| RFC | technical name | A document that gives a standard for networks. Each RFC has a number. |
| RMS norm | technical noun | A step of a model layer that changes the numbers of one token to the same level. It then multiplies them by model weights. |
| rollback | technical noun | The change of the layer state back to a snapshot. |
| roofline floor | technical noun | The sum of the times of the kernels of one pass or of one request. The time of a kernel is the larger of two times: the time of its memory work and the time to calculate. |
| root folder | technical noun | The folder that has all other files of the repository. |
| rotation | technical noun | The change of a tensor with a matrix that does not change. It turns each list of numbers of the tensor and does not change the length of the list. |
| rsync | technical name | The program that puts the changes of a folder into a folder on a different host. |
| rule | technical noun | One instruction of STE for the text. |
| run | technical noun | One time that a program does its work from start to stop. |
| run | technical verb | To start a program and let it work until it stops. |
| S | technical name | The short name of the stock build of the tag `b12004`. Arm S uses it. |
| sampler | technical noun | A program that reads the memory use, the power, and the clock of the GPU 4 times each second. It writes the values to a file. |
| SASS | technical name | The program code that an NVIDIA GPU runs. A tool can show it as text. |
| scalar gate | technical noun | The number of a GDN layer that sets how much of the layer state stays for the next token. It is one number and not a list of numbers. |
| scale | technical noun | A number that a group of quantized weights has. The program multiplies each integer of the group by the scale to get a model weight. |
| scan | technical noun | One time that the publication gate or gitleaks scans text. A scan gives a list of items or no item. |
| scan | technical verb | To read the files and the history of a repository and to find private data. |
| schedule | technical noun | A text file with one line for each run. The harness does the lines in the sequence of the file. |
| scheduler | technical noun | The part of ggml that gives the nodes of a graph to the backends. It also uploads the inputs of each part of the graph. |
| script | technical noun | A short program that a person writes in a file of text. |
| secret | technical noun | A text that gives access to a server. An access token and an access key are secrets. A person must not give a secret to other persons. |
| section | technical noun | A heading and the text below it until the next heading of the same level or a higher level. |
| seed | technical noun | The start number of a random sequence. |
| semicolon | technical noun | The mark `;` in a sentence. |
| sentence | technical noun | A group of words that ends with a period. |
| sequential equal split | technical noun | An equal split in which the numbers that name the sequences of one ubatch are in sequence. |
| server | technical noun | A program that waits for requests and sends back responses. |
| setup error | technical noun | An error that stops the publication gate before the scans. A tool, an environment variable, or a file is missing. |
| SHA | technical name | A group of functions that calculate a checksum. SHA-256 is one of them. |
| shared library | technical noun | A file with code that a program reads when it starts. More than one program can use it. Its name ends with `.so`. |
| shared memory | technical noun | A fast memory on an SM. The threads of one thread block use it together. |
| shared prefix | technical noun | The prefix that is the same in the prompts of two or more questions. |
| shifted run | technical noun | A run of Nsight Compute in which the profiled launches have the kernels of the list. They start a number of launches from the first launch of the list. |
| short request | technical noun | A request for a case with a short state. The state has less than 100 tokens. |
| SiLU | technical name | The function `x/(1+exp(-x))`. |
| skip | technical verb | To make no launch for a node or a kernel. The ablation skips the work of a part of the model. |
| slot assignment | technical noun | The list of the task slots that hold the questions of one request. Different lists can give different passes. |
| SM | technical noun | One part of the GPU. It runs the threads of a thread block. The `RTX 3060 Ti` has 38 SMs. |
| snapshot | technical noun | A copy of the layer state of a recurrent layer at one token. The memory keeps it. The program can then go back to that token. |
| softmax | technical noun | A function that changes logits to probabilities. The sum of the probabilities is 1. |
| specification | technical noun | The PDF that gives the rules and the dictionary of STE. |
| speculative decoding | technical noun | A method for faster decode. A draft writes the next tokens, and the model checks all of them in one pass. |
| spelling | technical noun | The letters of a word in sequence. |
| split KV cache | technical noun | A KV cache with one different part for each slot. The unified KV cache is not the same. |
| SSH | technical name | The program that gives you a terminal on a different host. The link is safe. |
| SSM state | technical noun | The part of the layer state of a Gated DeltaNet layer that is not the conv state. |
| stage | technical noun | A group of runs of the ablation. Stage 1 has one run for each arm. Stage 2 has more blocks for some arms. |
| stall | technical noun | A cycle in which a warp does not start an instruction. |
| stall reason | technical noun | The cause of a stall. Nsight Compute gives the stalls of each stall reason, for example `wait` or `long_scoreboard`. |
| stall sample | technical noun | A sample of Nsight Compute in which a warp waits and does not run an instruction. The sample shows the instruction at which the warp waits. |
| state | technical noun | The text of one case. The model reads it before it answers the questions. |
| STE | technical name | The short name of `Simplified Technical English`. ASD-STE100 gives its rules. |
| stock build | technical noun | A build of llama.cpp that does not have the changes of this project. |
| stock image | technical noun | The image of llama.cpp for CUDA that the llama.cpp project publishes. It has the programs of the stock build. |
| stream synchronization | technical noun | A step in which the host or one CUDA stream waits until the work of other CUDA streams stops. |
| stream-k | technical noun | A method of MMQ that divides the total work of one matrix multiply in equal parts between the thread blocks. |
| string | technical noun | A sequence of text in a program or in a data file. |
| submit | technical verb | To send the launches of one pass to the GPU. |
| suffix | technical noun | The tokens of a prompt after the shared prefix. They have one question. |
| summary | technical noun | A short section at the start of a document. It gives the primary facts. |
| SwiGLU | technical name | The GLU of the feed-forward layers of this model. It multiplies the SiLU of the first list by the second list. |
| switch point | technical noun | The number of columns in a batch at which llama.cpp changes from MMVQ to MMQ. Each GPU and each type of quantization can have a different switch point. |
| sync call | technical noun | A CUDA call that waits until the GPU stops its work. The CUDA function `cudaStreamSynchronize` makes a sync call. |
| syslog | technical noun | The log of the system of a host. Each line of a syslog has a time, a host name, a program, and a text. |
| system wait | technical noun | The time in which the decode thread waits in a function of the system, for example `pthread_cond_wait`. |
| table | technical noun | Data in rows and columns. |
| table header | technical noun | The first row of a table. It gives the name and the unit of each column. |
| tag | technical noun | A name that marks one commit in a repository. |
| tag message | technical noun | The text that a person writes with a tag. |
| Tailscale | technical name | A program that connects the hosts of a group of persons in one network. |
| target | technical noun | A part of the work that can make the server faster. Each target has one letter. |
| task slot | technical noun | The part of the server that holds the KV cache and the state of one task. The flag `-np` sets the number of task slots. |
| technical debt | technical noun | The work that a project must do after a change, because the change makes the code hard to keep. |
| temperature | technical noun | The heat of the GPU, in °C. |
| template | technical noun | A text with empty parts. A person writes the correct text in each empty part. |
| tensor | technical noun | A group of numbers with a shape. |
| tensor core | technical noun | The part of an SM that runs an MMA instruction. |
| term | technical noun | A word or a group of words that has one meaning in the project. |
| terminal | technical noun | The program in which you write commands. |
| text | technical noun | Words and marks that a person can read. |
| TFLOPS | technical name | The unit of 10^12 floating point operations for each second. |
| thread | technical noun | The smallest unit of work in a kernel. A warp has 32 threads. |
| thread block | technical noun | A group of threads that use the same SM at the same time. |
| threshold | technical noun | A limit for a result. We write it before the test. |
| throughput | technical noun | The number of decisions that the server finishes in one second. |
| tile | technical noun | A small part of a matrix that one thread block reads and writes at one time. |
| time model | technical noun | The functions of the script `make_tables.py` in the data set `2026-10-31-phase4-model`. They calculate the time of a request after a change, from the kernel times and the assumptions. |
| timed pass | technical noun | One time that the client sends all cases to the server, one request at a time, in a random sequence. The client measures the latency of each request. |
| title | technical noun | The text that gives a name to an issue or to a pull request. |
| TMA | technical name | The part of a Hopper GPU or of a Blackwell GPU that copies tiles between VRAM and shared memory. |
| token | technical noun | One piece of text that the model reads or writes. |
| tokenizer | technical noun | The program that changes text to tokens. |
| TOPS | technical name | The unit of 10^12 operations for each second. A multiply-add counts as 2 operations. |
| transpose | technical verb | To change the position of the rows and the columns of a matrix. The rows of the new matrix are the columns of the matrix. |
| true | technical name | The value of a condition that is correct. |
| TSV | technical name | A type of text file for data. In each line, the mark `\t` is between two values. |
| Turing | technical name | A type of NVIDIA GPU. It comes after Volta. |
| ubatch | technical noun | A part of a batch. The model reads one ubatch in each pass. The flag `-ub` sets its largest number of tokens. |
| unified KV cache | technical noun | One KV cache that all slots of the server use together. The flag `--kv-unified` selects it. |
| Unix time | technical noun | The time as a number of seconds from the date `1970-01-01 00:00 UTC`. |
| upload | technical noun | A copy of data from the memory of the CPU to the memory of the GPU. |
| upload | technical verb | To write data from the memory of the CPU to the memory of the GPU. |
| upstream base | technical noun | The revisions of upstream llama.cpp. The publication gate does not scan their history. |
| URL | technical name | The text that gives the position of a file or of a server on a network, for example `https://example.com/a`. |
| user | technical noun | A person or a program that uses a server or a host. |
| UTC | technical name | The standard of time for all dates and times of the project. |
| UUID | technical name | A code of 128 bits that gives a name to one item. |
| value-split path | technical noun | A path of the kernel `gdn_single`. Different thread blocks calculate different value columns of one head. |
| variant | technical noun | One time that the server gives a question to the model. The possible answers are in one sequence. The server can give a question more than one time, with a different sequence each time. It then calculates the mean of the probabilities. |
| vault item ID | technical noun | The UUID of one item in a program that keeps the secrets of a person. |
| VBIOS | technical name | The program code in the memory of the GPU that starts it. Its version is a text, for example `94.04.3A.40.A7`. |
| vectorize | technical verb | To change a kernel to make it read, write, or calculate more than one number with one instruction. |
| verb | technical noun | A word for a task that a person or a device does. |
| version | technical noun | The number or the name that shows which code of a program you have. A new code has a new number. |
| vocabulary | technical noun | The set of all tokens that the model knows. The model gives one logit for each token of the vocabulary. |
| Volta | technical name | A type of NVIDIA GPU. |
| VRAM | technical noun | The memory on the GPU. |
| Vulkan | technical name | A system that lets a program use a GPU. It is not CUDA. |
| wall time | technical noun | The time from the start of a request at the client to the end of the response at the client. |
| warning | technical noun | A finding that is not an error. It does not change the exit code. |
| warp | technical noun | A group of 32 threads that do the same step at the same time. |
| warp divergence | technical noun | The condition when the threads of one warp do different steps of the code. |
| WebUI | technical name | The interface of llama-server that a person can use to send text to the model. The flag `--no-webui` turns it off. |
| week | technical noun | A unit of time. One week has 7 days. |
| weight floor | technical noun | The sum of the times of the passes of one request. The time of a pass is the larger of two times: the time to read the weights from the DRAM, and the time of the int8 operations. |
| weight-read floor | technical noun | The time that the GPU uses to read all weights of the model from the DRAM at the best DRAM read rate. No pass is faster than it. |
| whole-head path | technical noun | A path of the kernel `gdn_single`. One thread block calculates all value columns of one head. |
| word | technical noun | One unit of text between two spaces. |
| wrapper | technical noun | A program that starts a second program. It records or checks the work of the second program. |
