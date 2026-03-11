# Agent Tool Benchmark for Predictive Analytics
Predictive Analytics Group Coursework - MSIN0097

This project evaluates the performance of agentic tools on realistic data science tasks.

## Objective

The goal is to benchmark several agent tools on common data science workflows and evaluate them on:
- correctness
- statistical validity
- reproducibility
- code quality
- efficiency

## Tools Compared

- Claude Code
- Codex
- Antigravity

## Benchmark Tasks

1. Dataset ingestion and validation
2. Exploratory data analysis
3. Baseline model training
4. Data leakage detection

## Dataset  
The experiments use the Airline Passenger Satisfaction dataset available on Kaggle (https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data). The dataset contains survey responses from airline passengers along with demographic information, flight characteristics, and service quality ratings. The target variable indicates whether a passenger was satisfied or dissatisfied with their flight experience.

The dataset includes both categorical and numerical features such as travel class, flight distance, service ratings (e.g., seat comfort, onboard service), and delays. This makes it well suited for testing data science workflows including data ingestion, exploratory analysis, and classification modelling.

The objective of the predictive task is to build a model that predicts passenger satisfaction based on these features.

## Repository Structure

- task/ -> benchmark task specifications
- notebooks/ -> experimental notebooks
- results/ -> outputs from each agent
- figures/ -> plots used in the report
- appendix/ -> logs & prompts
