# ail-homework
The homework for my Artificial Intelligence Laboratory course.  
  
This repository contains my homework submission for the [Artificial Intelligence Laboratory](https://moodle.uni-pannon.hu/course/view.php?id=24556) course.
  
## Description  
  
I have received the following instructions to complete as the homework:  
* Make use of a Large Language Model in a complex way, utilizing programming and some form of data transformation.  
  
Additional guidance I've received:  
Don't just prompt a model, then simply pass its output into another model. Do some form of transformation on it first. The submission should implement some more complex functionality. Utilizing multiple modalities is also encouraged.  
  
I will have to present my results, which can be just running and explaining the code, or a presentation and a recording. (Recommended in the case of longer runtimes.)  
I will have to transfer useful knowledge and call out the pitfalls I've encountered.
  
## My idea  
  
My idea is to implement a Video Content Analysis solution.  
The outline of which is described as follows:  
  
1. Take a video file as input, transcribe it utilizing a model, resulting in a timestamped transcript.
2. Take the transcript, chunk/segment it into 'logical chapters' based on the topic changes in the content.
3. Summarize, keyword extract the chapters.
4. Embed the summaries, keywords, store in a local VectorDB. (e.g.:FAISS)
5. Get a query from the user, embed it, retrieve relevant chapters from the DB (possibly with timestamps?).
6. Answer the user's query using the provided context.
