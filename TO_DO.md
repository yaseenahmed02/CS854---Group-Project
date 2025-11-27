# Doing:

# To Do: 

- [ ] if the instance has no images, skip it. (Dev w/ image 102/102; Test w/ image 462/510)

- [ ] better understand how the split for files bigger than 8192 tokens during the repo ingestion step works. 

- [ ] Implement a strategy for top-k retrieval, where the user can define the k value. If the resulting ammount of tokens is less than the total token limit, then it should retrieve only the top-k documents. If the resulting ammount of tokens is greater than the total token limit, then it should retrieve the top-k documents and remove the least relevant documents until the limit is reached. 
- [ ] Implement a strategy not top-k, where the user can define the total token limit. If the resulting ammount of tokens is still less than the total token limit, then it should retrieve more relevant documents. 


# Done: 
- [x] send the image file to the LLM, not just the retrieved context. 
- [x] make image downloading and storage mandatory during ingestion. 
- [x] Improve the current `run_pipeline.py` strategy to allow for a more responsive way to process the SWE-bench dataset. It should start by asking which split, then presenting all the available repos for that split, with a very short description of the repo project, and then present the versions of the repo and the total number of instances for each version, and then allow the user to select which one(s) to process. It should also allow the user to define the total token limit for the retrieval step.  It should also ask which LLM to use for the VLM image description generation step and for the final LLM generation step (same LLM for both), and what it the maximim output token limit for the final LLM generation step. This could be a series of questions, or a single question with multiple options. Maybe in the console, maybe another way. Consider what is the simplest way to "build"  this command. The current command is too complex and not user-friendly, with too many parameters. Present the plan and ask before starting.
- [x] Check any places where token limits are hard coded and consider how it influences.
- [x] Create a metrics file to store the metrics for each instance processed. It should store the time taken to generate the image description, the ammount of tokens in the textual part of the instance, the ammount of images of the instance, the ammount of tokens in the image description (if more than one image, sum it up), the token limit for that run, the ammount of tokens in the retrieved content, the time it took to retrieve the context, the total ammount of tokens in the input to the LLM, and the final solution output token count.
- [x] Refactor the final LLM solution generation step to use GPT-4o. Leave a parameter to allow for the use of a mock LLM, and the model or mock shall be defined in the `run_pipeline.py` script. Also leave the parameter to run it against a model running on vLLM, so the prompt will be sent to vLLM via its API. 
- [x] Create a metrics file to store the metrics for each repo/version processed. It should store the time taken to embed, the repo size, the final vector db size, and other related metrics.
- [x] Add a parameter for limiting the total tokens in the input to a given multiple of thousands of tokens (e.g. 13, 27, 50) for the final LLM generation step. It should start by counting the ammount of tokens in the textual part of the GitHub issue, then the ammount of tokens in the image(s) description(s), if any, and then top up with retrieved content up to the given multiple of thousands of tokens, to be sent to the LLM for the generation step. 
- [x] Refactor the VLM image description generation step to use GPT-4o, but leave a parameter to allow for the use of a mock VLM, and the model or mock shall be defined in the `run_pipeline.py` script.
- [x] Implement a strategy in which I can pinpoint the repo/version of the SWE-bench dataset that I want to be processed, and it should retrieve all instances from that repo/version, not only the hard-coded markedjs/marked 1.2.
- [x] Add the repo version to the data/repo folder name... can it?
- [x] Allow it to retrieve multiple images (up to 10) from the same instance.
- [x] Implement a token counter/limit for the retrieval step, to create a condition comparable to the original SWE-bench RAG experiment, that used 13k/27k/50k tokens for the retrieval step.
- [x] Refactor test and verification strategy.