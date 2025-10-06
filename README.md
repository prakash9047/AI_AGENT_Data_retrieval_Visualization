# Natural Language to SQL Query Agent

This project is a Flask-based web application that allows users to ask natural language questions, which are then converted into SQL queries to fetch data from a database. The results are analyzed and visualized using Plotly charts, providing an interactive and user-friendly experience.

---

## Features

- **Natural Language to SQL Conversion**: Converts user questions into SQL queries using an AI-powered agent.
- **Database Integration**: Connects to a MySQL database to fetch data.
- **Data Visualization**: Automatically generates visualizations (bar charts, pie charts, line charts, etc.) based on query results.
- **Error Analysis**: Provides insights and suggestions when a query fails.
- **REST API**: Exposes an API endpoint (`/analyze`) for programmatic access.

---

## Installation

1. Clone the repository:
   ```bash
   git clone "link of folder"
   cd folder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```env
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_NAME=your_database_name
   ```

4. Run the application:
   ```bash
   python app.py
   ```

5. Access the application in your browser at `http://127.0.0.1:5000`.

---

## API Usage

### Endpoint: `/analyze`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "question": "What is the average salary by department?"
  }
  ```
- **Response**:
  ```json
  {
    "raw_output": "SQL query result or error message",
    "analysis_result": {
      "has_data": true,
      "data_sets": [
        {
          "title": "Average Salary by Department",
          "chart_type": "bar",
          "data": {
            "columns": ["Department", "Average Salary"],
            "rows": [["HR", 50000], ["Engineering", 80000]]
          }
        }
      ],
      "analysis": "Explanation of the query result",
      "suggestions": ["Alternative question 1", "Alternative question 2"]
    },
    "visualizations": [
      {
        "type": "bar",
        "title": "Average Salary by Department",
        "data": {
          "columns": ["Department", "Average Salary"],
          "rows": [["HR", 50000], ["Engineering", 80000]]
        }
      }
    ],
    "schema_analysis": null
  }
  ```

---

## Project Structure

```
.
├── app.py                 # Main Flask application
├── templates/             # HTML templates for the web interface
├── static/                # Static files (CSS, JS, etc.)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
└── README.md              # Project documentation
```

---



## Technologies Used

- **Backend**: Flask, SQLAlchemy
- **AI Agent**: LangChain, OpenAI GPT-4o
- **Database**: MySQL
- **Visualization**: Plotly
- **Frontend**: HTML, CSS, JavaScript

---

## Future Enhancements

- Add support for more database types (e.g., PostgreSQL, SQLite).
- Improve error handling and analysis.
- Add user authentication for secure access.
- Enhance visualization options with more chart types.



- [LangChain](https://github.com/langchain-ai/langchain)
- [OpenAI](https://openai.com/)
- [Plotly](https://plotly.com/)
