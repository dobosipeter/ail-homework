# Pitfalls  
  
This document describes the the pitfalls that I've encountered, and how I've mitigated them.  
  
__LLM Context Rot__:  I've made use of an LLM in various parts of the solution process from ideation, through planning to implementation.  
I've been using Google Gemini 3 Pro with Thinking. My workflow was having the model open in a browser window, and I've provided it with the git repository containing my work on GitHub as context, through the import code feature.  
Despite this, and keeping my entire conversation in a single chat, I've constantly had to repeat myself and remind it of specific nuances that I've requested to be considered already earlier.  
  
For example:  
  
✨: Extracting the audio to FLAC is preferable, because it avoids compression artifacts that plauge MP3s, inducing hallucinations in transcription models.  
👤: Great, so let's draft an initial implementation!  
✨: So here is my implementation where I extract the audio to MP3.  
👤: But didn't you just say that FLAC is preferable?  
✨: _You are absolutely correct!_ Here is an implementation using FLAC.  
👤: ...  


__Outdated LLM Info__: This one might be specific to the model I've used, but it had a hard time providing me with up to date information throughout my work. It kept suggesting implementations, type hints and dependency versions that were already outdated.  
  
__Low quality suggestions__: The quality of the implementation suggestions were a mixed bag. For example initially it used prints instead of actual logging. Then after I've requested it to use the logging module, it implemented every single log call with an f-string instead of using lazy logging.

__Overly verbose & cheerful__:  
* _You are absolutely right..._
* _This is a sophisticated request that moves the project from "Student Homework" to "Production Engineering."_ -- (After I've requested it to use logging instead of prints...)
* _This isn't just <something simple>... This is <something more impressive>!_
* _\# This display the results_ -> _result.display()_  
  
Mitigated the above by manually reviewing each function before implementing, removing useless comments, fixing other mistakes.  
  
__API Call Failures__: My calls to the transcription API failed seemingly randomly, had to implement retries, backoff. Maybe caused by some network issues?
  
__Dependency Hell__: I've created a new venv for the project. Defined nothing but streamlit as a dependency. Attempted to install the package failed, because pip tried to build one of the dependencies of streamlit, which failed. Apparently this is relatively common issue, something about the interplay between recent python releases and dependency versions.  
Solved it by pinning an older streamlit version, and downgrading the venv to 3.12.  

## Positives  
* Much faster than doing everthing manually.
* A decent starting point.
* Lots of fun to iterate with and to see the first parts pop-up in the first hour.   

