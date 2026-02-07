from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()
import os
from typing import List, Optional,Annotated,TypedDict
from langgraph.graph.state import LastValue
import operator

groq_api_key=os.getenv("GROQ_API_KEY")

class EvalutionSchema(BaseModel):
    feedback:str=Field(description="Feedback for the essay")
    score:int=Field(description="Score out of 10 for the essay",ge=0,le=10)

model=ChatGroq(model="qwen/qwen3-32b", api_key=groq_api_key)

structured_model=model.with_structured_output(EvalutionSchema)

#********************************** essay ***********************************

essay="""India and AI Time

Now world change very fast because new tech call Artificial Intel… something (AI). India also want become big in this AI thing. If work hard, India can go top. But if no careful, India go back.

India have many good. We have smart student, many engine-ear, and good IT peoples. Big company like TCS, Infosys, Wipro already use AI. Government also do program “AI for All”. It want AI in farm, doctor place, school and transport.

In farm, AI help farmer know when to put seed, when rain come, how stop bug. In health, AI help doctor see sick early. In school, AI help student learn good. Government office use AI to find bad people and work fast.

But problem come also. First is many villager no have phone or internet. So AI not help them. Second, many people lose job because AI and machine do work. Poor people get more bad.

One more big problem is privacy. AI need big big data. Who take care? India still make data rule. If no strong rule, AI do bad.

India must all people together – govern, school, company and normal people. We teach AI and make sure AI not bad. Also talk to other country and learn from them.

If India use AI good way, we become strong, help poor and make better life. But if only rich use AI, and poor no get, then big bad thing happen.

So, in short, AI time in India have many hope and many danger. We must go right road. AI must help all people, not only some. Then India grow big and world say "good job India".
"""

#****************************** *************************************************






class UPSCState(TypedDict):
    essay:Annotated[str,LastValue]
    language_feedback:Annotated[str,LastValue]
    analysis_feedback:Annotated[str,LastValue]
    clarity_feedback:Annotated[str,LastValue]
    overall_feedback:Annotated[str,LastValue]
    individual_scores:Annotated[List[int],operator.add]
    avg_score:float

@traceable(name="language_evalution_fn",tags=['dimnesion:language'],metadata={"dimension":"language"})
def language_evalution_fn(state:UPSCState):
    prompt=PromptTemplate.from_template(
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n{essay}"
    )
    out=structured_model.invoke(prompt.format(essay=state['essay']))

    feedback=out.model_dump()['feedback']
    score=out.model_dump()['score']
    return {"language_feedback":feedback,"individual_scores":[score]}

# langusge=language_evalution_fn({"essay":essay})
# print(langusge['language_feedback'])

@traceable(name="anaylsis_evalution_fn",tags=['dimnesion:analysis'],metadata={"dimension":"analysis"})
def analysis_evalution_fn(state:UPSCState):
    prompt=PromptTemplate.from_template(
        """
    Evaluate the depth of analysis of the following essay and provide feedback
    and assign a score out of 10.\n\n{essay}"""
    )
    out=structured_model.invoke(prompt.format(essay=state['essay']))

    feedback=out.model_dump()['feedback']
    score=out.model_dump()['score']

    return {"analysis_feedback":feedback,"individual_scores":[score]}

# anaylis=analysis_evalution_fn({"essay":essay})

@traceable(name="clarity_evalution_fn",tags=['dimnesion:clarity'],metadata={"dimension":"clarity_of_thought"})
def clarity_evalution_fn(state:UPSCState):
    prompt=PromptTemplate.from_template(
    """
    Evaluate the clarity of thought of the following essay and provide feedback
    and assign a score out of 10.\n\n{essay}"""
    )

    out=structured_model.invoke(prompt.format(essay=state['essay']))

    feedback=out.model_dump()['feedback']
    score=out.model_dump()['score']
    return {"clarity_feedback":feedback,"individual_scores":[score]}

# clarity=clarity_evalution_fn({"essay":essay})
# print(clarity['clarity_feedback'])

@traceable(name="final_evalution_fn",tags=['aggregate'])
def final_evalution_fn(state:UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

graph=StateGraph(UPSCState)

graph.add_node("language_evalution",language_evalution_fn)
graph.add_node("analysis_evalution",analysis_evalution_fn)  
graph.add_node("clarity_evalution",clarity_evalution_fn)
graph.add_node("final_evalution",final_evalution_fn)


graph.add_edge(START,"language_evalution")
graph.add_edge(START,"analysis_evalution")
graph.add_edge(START,"clarity_evalution")
graph.add_edge("language_evalution","final_evalution")
graph.add_edge("analysis_evalution","final_evalution")
graph.add_edge("clarity_evalution","final_evalution")
graph.add_edge("final_evalution",END)

workflow=graph.compile()

# ---------- Direct invoke without wrapper ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay},
        config={
            "run_name": "evaluate_upsc_essay",  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay),
                "model": "qwwen3-32b",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print(result)
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))

