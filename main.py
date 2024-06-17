from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, FileReadTool, DirectoryReadTool, ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()

web_search = SerperDevTool()
file_read = FileReadTool()
file_search =  DirectoryReadTool()
web_scrape = ScrapeWebsiteTool()

# create agents

pesquisador = Agent(
    role = 'Pesquisador', 
    goal='Pesquisar na internet as últimas 2 noticias sobre IA',
    backstory='Especialista em realizar pesquisa na Internet',
    model='gpt-4o',
    verbose=True,
    tools=[web_search]

)

escritor = Agent(
    role = 'Escritor', 
    goal='Escolher apenas 1 notícias e realizar um resumo do conteúdo',
    backstory='Especialista em escolher apenas 1 notícia',
    model='gpt-4o',
    verbose=True,
    tools=[web_search,web_scrape, file_search, file_read]

)


#Tasks

pesquisa = Task(
    description='Pesquisar na internet notícias impactantes sobre IA',
    expected_output='Uma lista com 3 URLs das IA mais recentes',
    agent=pesquisador,
    tools=[web_search],
   

)

resumo = Task(
    description='Escolher um dos temas e fazer o resumo sobre',
    expected_output='O resumo apenas de uma notícia escolhida dentre as 3 notícias que foram coletadas',
    agent=escritor,
    tools=[web_search,web_scrape, file_search, file_read],
    output_file='resumo.txt'

)

crew = Crew(
    agents=[pesquisador, escritor],
    tasks=[pesquisa, resumo],
    process=Process.sequential,
    verbose=2

)

crew.kickoff()

#python -X utf8 main.py
