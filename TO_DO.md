- [x] Allow it to retrieve multiple images (up to 10) from the same instance.
- [ ] Implement a token counter/limit for the retrieval step, to create a condition comparable to the original SWE-bench RAG experiment, that used 13k/27k/50k tokens for the retrieval step.
- [ ] Check token limits (input and output) for the VLM generation step. 
- [ ] Check any places where token limits are hard coded and consider how it influences.
- [ ] Implement a strategy in which I can pinpoint the repo/version of the SWE-bench dataset that I want to be processed, and it should retrieve all instances from that repo/version.
```
repo version total_instances
Automattic/wp-calypso 10.15.2 10
Automattic/wp-calypso 8.9.2003 9
chartjs/Chart.js 3.0 9
diegomura/react-pdf 2.0 9
Automattic/wp-calypso 8.9.2004 5
chartjs/Chart.js 3.5 4
processing/p5.js 1.5 4
Automattic/wp-calypso 10.6.2000 3
markedjs/marked 1.2 3
processing/p5.js 1.4 3
``` 