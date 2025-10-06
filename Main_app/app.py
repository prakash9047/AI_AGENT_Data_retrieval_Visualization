# app.py

import os
import json
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from flask import Flask, request, jsonify, render_template

# --- Initialization ---
load_dotenv()
app = Flask(__name__, template_folder='templates')

# --- All your original helper functions go here (with minor adjustments) ---

# Caching is handled differently in Flask; for simplicity, we'll initialize these once.
# For production, consider using Flask-Caching or a similar library.
_engine = None
_sql_agent = None
_llm = None
_schema_info = None
_tables = None

def get_db_engine():
    global _engine
    if _engine is None:
        try:
            user = os.getenv("DB_USER")
            raw_password = os.getenv("DB_PASSWORD")
            password = quote_plus(raw_password)
            host = os.getenv("DB_HOST")
            dbname = os.getenv("DB_NAME")
            db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}/{dbname}"
            _engine = create_engine(db_uri)
            # Test connection
            with _engine.connect() as conn:
                print("Database Connected Successfully")
        except Exception as e:
            print(f"DB Connection Failed: {e}")
            _engine = None
    return _engine

def get_database_schema():
    global _schema_info, _tables
    if _schema_info is None or _tables is None:
        try:
            engine = get_db_engine()
            if not engine: return {}, []
            with engine.connect() as conn:
                tables_result = conn.execute(text("SHOW TABLES"))
                _tables = [row[0] for row in tables_result]
                
                _schema_info = {}
                for table in _tables:
                    columns_result = conn.execute(text(f"DESCRIBE {table}"))
                    columns = [{'field': row[0], 'type': row[1]} for row in columns_result]
                    _schema_info[table] = columns
        except Exception as e:
            print(f"Error getting schema: {e}")
            return {}, []
    return _schema_info, _tables


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm

def get_sql_agent():
    global _sql_agent
    if _sql_agent is None:
        engine = get_db_engine()
        if not engine: return None
        try:
            db = SQLDatabase(engine=engine)
            llm = get_llm()
            _sql_agent = create_sql_agent(
                llm=llm,
                db=db,
                agent_type="openai-tools",
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"Failed to initialize AI Agent. Error: {e}")
            return None
    return _sql_agent

def analyze_query_failure(user_question, sql_response, schema_info):
    llm = get_llm()
    schema_str = "\n".join(
        f"Table: {table}\n" + "".join(f"  - {col['field']} ({col['type']})\n" for col in columns)
        for table, columns in schema_info.items()
    )
    
    prompt = f"""
    A user asked: "{user_question}". The SQL agent responded: "{sql_response}".
    DATABASE SCHEMA:
    {schema_str}
    Analyze why the query failed and provide:
    1. A likely reason for the failure (e.g., data doesn't exist, wrong column names).
    2. Alternative questions based on the actual schema.
    3. Sample queries that would work.
    Return your response in this exact JSON format:
    {{
        "analysis": "...",
        "alternative_questions": ["...", "..."],
        "sample_queries": ["...", "..."],
        "schema_limitations": "..."
    }}
    """
    try:
        response = llm.invoke(prompt)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        return {"analysis": f"Error in analysis: {e}", "alternative_questions": [], "sample_queries": []}
    return {"analysis": "Could not analyze the failure.", "alternative_questions": [], "sample_queries": []}


def extract_and_visualize_data(text_response, user_question):
    llm = get_llm()
    prompt = f"""
    Analyze this database response and user question. Extract structured data for visualization.
    USER QUESTION: {user_question}
    QUERY RESPONSE: {text_response}
    TASK:
    - If the response contains structured data, extract it.
    - If no data is found, explain why.
    - Suggest alternative questions.
    OUTPUT FORMAT (exact JSON):
    {{
        "has_data": true/false,
        "data_sets": [
            {{
                "title": "Chart title",
                "chart_type": "bar|pie|line|table",
                "data": {{"columns": ["col1", "col2"], "rows": [["val1", 100]]}}
            }}
        ],
        "analysis": "Explanation of what was found or not found.",
        "suggestions": ["Suggestion 1", "Suggestion 2"]
    }}
    """
    try:
        response = llm.invoke(prompt)
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        return {"has_data": False, "data_sets": [], "analysis": f"Error: {e}", "suggestions": []}
    return {"has_data": False, "data_sets": [], "analysis": "Could not parse response.", "suggestions": []}

def create_visualization_json(data_set):
    """MODIFIED: Create visualization and return it as JSON for Plotly.js"""
    try:
        df = pd.DataFrame(data_set['data']['rows'], columns=data_set['data']['columns'])
        
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('$', '').str.replace(',', ''), errors='ignore')
        
        chart_type = data_set.get('chart_type', 'bar')
        title = data_set.get('title', 'Visualization')
        
        fig = None
        if chart_type == 'bar' and len(df.columns) >= 2:
            fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=title)
        elif chart_type == 'pie' and len(df.columns) >= 2:
            fig = px.pie(df, names=df.columns[0], values=df.columns[1], title=title)
        elif chart_type == 'line' and len(df.columns) >= 2:
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title)
        
        if fig:
            return json.loads(fig.to_json()) # Return JSON string
        else: # Fallback to table data
            return {
                "type": "table",
                "title": title,
                "header": list(df.columns),
                "cells": [df[col].tolist() for col in df.columns]
            }
            
    except Exception as e:
        print(f"Error creating visualization JSON: {e}")
        return None

# --- Flask Routes ---

@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """API endpoint to process the user's question."""
    data = request.json
    user_question = data.get('question')

    if not user_question:
        return jsonify({"error": "Please provide a question."}), 400

    agent = get_sql_agent()
    schema_info, _ = get_database_schema()

    if not agent:
        return jsonify({"error": "AI Agent could not be initialized. Check DB connection."}), 500

    try:
        # Step 1: Query database
        result = agent.invoke({"input": user_question})
        output = result.get("output", "No result found.")

        # Step 2: Analyze the response for data
        analysis_result = extract_and_visualize_data(output, user_question)
        
        response_data = {
            "raw_output": output,
            "analysis_result": analysis_result,
            "visualizations": [],
            "schema_analysis": None
        }

        if analysis_result.get('has_data') and analysis_result.get('data_sets'):
            for data_set in analysis_result['data_sets']:
                vis_json = create_visualization_json(data_set)
                if vis_json:
                    response_data["visualizations"].append(vis_json)
        else:
            # If no data, run the failure analysis
            response_data["schema_analysis"] = analyze_query_failure(user_question, output, schema_info)
        
        return jsonify(response_data)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    get_db_engine() # Pre-connect to the database on startup
    get_database_schema()
    app.run(debug=True)