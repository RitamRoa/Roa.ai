# my own ai (well took a bit help for folder structure cuz i am still learning..

basically learnt on api calling on how you can efficietly use it my roa.py file is perfect for that. Used ai for setting up flask and frontend
i have hosted backend in google cloud, link : https://roa-expert-system-115413225766.asia-south1.run.app
just open the index.html, wont work if you open up the above link. Used ai for setting up dockerfile cuz i knew too less about it.

## what?

so it is more of a fact-checker, the thing is right now it might be very vulnerable to prompt injections, i need to fix that up.
as of 16th august: now should work if you open up the index

## how?

1.  clone this repo
2.  python -m venv .venv
3.  source .venv/bin/activate (remove "source" if you arent on mac)

### requirements:

i would suggest you to install these packages :
langchain-google-genai==0.0.1
langchain==0.2.1
langgraph==0.0.60
pydantic==2.8.2
python-dotenv==1.0.1
google-generativeai==0.7.0
