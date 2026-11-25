from typing import TypedDict, Literal
from pydantic import BaseModel
from common import model
from registry import warranty
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import TextContent
class State(TypedDict, total=False):
    question:str
    image_ref:str
    category:str
    answer:str
class Route(BaseModel):
    category:Literal['documentation','garantie']
@tool
def consulter_garantie(serial: str) -> dict:
    """Consulte en lecture seule le registre fictif par numéro de série."""
    return warranty(serial)
async def orchestrateur(state):
    if state.get('answer'): return {}
    route=await model().with_structured_output(Route, method='function_calling').ainvoke([
        ('system', 'Classe la demande en une seule catégorie. Garantie pour une vérification de couverture ou un '
         'numéro de série seul. Documentation pour un dépannage ou une question sur un appareil. '
         'Si la demande combine les deux, choisis le besoin principal.'),
        ('user', state['question']),
    ])
    return {'category':route.category}
def direction(state):
    return END if state.get('answer') else 'agent_'+state['category']
async def agent_documentation(state):
    client=MultiServerMCPClient({'sav':{'transport':'streamable_http','url':'http://127.0.0.1:23000/mcp'}})
    async with client.session('sav') as session:
        result = await session.call_tool('rag_multimodal', {
            'question': state['question'],
            'image_ref': state.get('image_ref', ''),
        })
    if result.isError:
        raise RuntimeError('Le serveur MCP SAV a signalé une erreur.')
    answer = '\n'.join(block.text for block in result.content if isinstance(block, TextContent)).strip()
    return {'answer': answer or 'Le serveur SAV n’a retourné aucune réponse.'}
async def agent_garantie(state):
    agent=create_agent(model(),[consulter_garantie],system_prompt='Consulte le registre avec le numéro de série. Si absent, demande-le. Ne devine pas.')
    r=await agent.ainvoke({'messages':[('user',state['question'])]})
    return {'answer':str(r['messages'][-1].content)}
g=StateGraph(State)
g.add_node('orchestrateur',orchestrateur);g.add_node('agent_documentaire',agent_documentation);g.add_node('agent_garantie',agent_garantie)
g.add_edge(START,'orchestrateur')
g.add_conditional_edges('orchestrateur',direction,{'agent_documentation':'agent_documentaire','agent_garantie':'agent_garantie',END:END})
for n in ['agent_documentaire','agent_garantie']:g.add_edge(n,'orchestrateur')
graph=g.compile()
