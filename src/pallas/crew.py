import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class Pallas:
    """Pallas crew"""

    baseUrl = os.getenv("BASE_URL")
    webSearchUrl = os.getenv("WEB_SEARCH_URL")
    apiKey = os.getenv("OPENAI_API_KEY")

    gemma2_9b = LLM(model="gemma2:9b", base_url=baseUrl, api_key=apiKey)

    mistral_nemo_12b = LLM(
        model="mistral-nemo:latest",
        base_url=baseUrl,
        api_key=apiKey,
        temperature=0.5,
    )
    llama3_2_3b = LLM(
        model="llama3.2:3b-instruct-q4_K_M",
        base_url=baseUrl,
        api_key=apiKey,
    )
    deepseek_r1_8b = LLM(
        model="deepseek-r1:8b",
        base_url=baseUrl,
        api_key=apiKey,
        temperature=0.5,
    )

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def interrogator(self) -> Agent:
        return Agent(
            config=self.agents_config["interrogator"],  # type: ignore[index]
            verbose=True,
            llm=self.gemma2_9b,
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config["manager"],  # type: ignore[index]
            verbose=True,
            allow_delegation=True,
            llm=self.mistral_nemo_12b,
        )

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            llm=self.llama3_2_3b,
        )

    @agent
    def reasoner(self) -> Agent:
        return Agent(
            config=self.agents_config["reasoner"],  # type: ignore[index]
            verbose=True,
            llm=self.deepseek_r1_8b,
        )

    @agent
    def reviser(self) -> Agent:
        return Agent(
            config=self.agents_config["reviser"],  # type: ignore[index]
            verbose=True,
            llm=self.gemma2_9b,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def interrogation_task(self) -> Task:
        return Task(
            config=self.tasks_config["interrogation_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @task
    def strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["strategy_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def reasoning_task(self) -> Task:
        return Task(
            config=self.tasks_config["reasoning_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @task
    def revision_task(self) -> Task:
        return Task(
            config=self.tasks_config["revision_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Pallas crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,
            manager_agent=self.manager,  # type: ignore
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
        )
