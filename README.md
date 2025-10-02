# ClimateScope: Visualizing Global Weather Trends and Extreme Events

## Project Objective
The objective of ClimateScope is to analyze and visually represent global weather patterns using the Global Weather Repository dataset. This project aims to uncover seasonal trends, regional variations, and extreme weather events through interactive and insightful visualizations. By leveraging daily-updated, worldwide weather data, the project will enable users to explore climate behavior over time, compare conditions across regions, and identify anomalies. The ultimate goal is to provide an accessible, data-driven platform that supports climate awareness, decision-making, and further research into global weather dynamics.

## Project Workflow

### Data Acquisition
- Download the Global Weather Repository dataset from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data).
- Ensure daily updates are synchronized if using the live version.

### Data Understanding & Exploration
- Inspect dataset structure, data types, and key variables (temperature, humidity, precipitation, wind speed, etc.).
- Identify missing values, anomalies, and coverage across regions.

### Data Cleaning & Preprocessing
- Handle missing or inconsistent entries.
- Convert units (if necessary) and normalize values.
- Aggregate or filter data (e.g., daily → monthly averages) for efficiency.

### Data Analysis
- Perform statistical analysis to understand distributions and correlations.
- Identify seasonal patterns, trends, and extreme weather events.
- Compare weather conditions across countries, continents, or climate zones.

### Data Visualization Design
- Select suitable visualization types:
  - Choropleth maps for geographic patterns  
  - Line charts for time-series trends  
  - Scatterplots for correlations  
  - Heatmaps for seasonal variation  
- Design an interactive dashboard layout.

### Visualization Development
- Build static and interactive visualizations using tools like Plotly, Tableau, Power BI, or D3.js.
- Integrate filters, sliders, and region selectors for user interactivity.

### Insights Generation
- Highlight notable findings such as temperature anomalies, high-precipitation zones, or wind speed extremes.
- Summarize regional and global climate trends.

### Final Dashboard & Reporting
- Deploy the dashboard as a web app or presentation.
- Document methodology, visualizations, and insights in a project report.

### Future Enhancements (Optional)
- Incorporate live API-based weather updates.
- Add predictive modeling for weather forecasting.
- Enable anomaly alerts for extreme events.

## Architecture Diagram
<img width="809" height="396" alt="image" src="https://github.com/user-attachments/assets/7a13767a-aab9-4010-8059-b952b9dd51a3" />


## Tech Stack
1. **Programming Language**
   - Python 3 – for data processing and visualization  
2. **Data Handling**
   - pandas – clean and transform data  
   - kaggle API – download dataset  
3. **Visualization**
   - Plotly – create interactive charts and maps  
   - Streamlit – build interactive dashboard  
4. **Storage**
   - CSV / Parquet files – store raw and processed data locally  
5. **Optional Tools**
   - folium – for additional map visualizations  
   - python-dotenv – manage API keys securely  

## Milestones

### Milestone 1: Data Preparation & Initial Analysis (Weeks 1-2)
**Tasks:**
- Download the Global Weather Repository dataset from Kaggle.  
- Set up project environment.  
- Inspect dataset structure, data types, and key variables.  
- Identify missing values, anomalies, and data coverage.  
- Handle missing or inconsistent entries.  
- Convert units and normalize values.  
- Aggregate or filter data (e.g., daily to monthly averages).  

**Evaluation:**
- **Deliverable:** Cleaned and preprocessed dataset, along with a summary document outlining data schema, key variables, and data quality issues.  
- **Success Criteria:** Dataset is successfully downloaded, cleaned, and transformed into a usable format, ready for analysis.
- <img width="927" height="552" alt="image" src="https://github.com/user-attachments/assets/349d89a2-969c-4c1e-973c-3ec0fa99b91e" />
<img width="913" height="771" alt="image" src="https://github.com/user-attachments/assets/f671cd4c-286d-4cd0-98d9-04afa54e2fb8" />
 
---

### Milestone 2: Core Analysis & Visualization Design (Weeks 2-4)
**Tasks:**
- Perform statistical analysis to understand distributions, correlations, seasonal patterns, and trends.  
- Identify extreme weather events.  
- Compare weather conditions across regions.  
- Select suitable visualization types (Choropleth maps, Line charts, Scatterplots, Heatmaps).  
- Design an interactive dashboard layout (wireframes/mockups).  

**Evaluation:**
- **Deliverable:** A report detailing analytical findings (statistical summaries, trends, extreme events, comparative analysis) and a set of wireframes/mockups for the dashboard.  
- **Success Criteria:** Clear analytical insights are derived and documented. Dashboard design effectively communicates planned insights with appropriate visualization choices.  
<img width="893" height="290" alt="image" src="https://github.com/user-attachments/assets/585f1f99-8964-404f-9817-a15ae4278b72" />
<img width="886" height="589" alt="image" src="https://github.com/user-attachments/assets/712f7530-07b2-4389-a423-abbac6b0658a" />
<img width="831" height="484" alt="image" src="https://github.com/user-attachments/assets/bf0448a4-ff43-4bd6-840a-70e7658e36d8" />
<img width="863" height="536" alt="image" src="https://github.com/user-attachments/assets/97b47778-53eb-4404-854c-c8ca6f9aaf22" />
<img width="407" height="365" alt="image" src="https://github.com/user-attachments/assets/c3f0c3f0-b3b2-4539-89ae-2b240cea4487" />
<img width="490" height="287" alt="image" src="https://github.com/user-attachments/assets/d2fc274c-dd38-4803-9249-a6e238af5788" />


---

### Milestone 3: Visualization Development & Interactivity (Weeks 4-6)
**Tasks:**
- Build static and interactive visualizations using chosen tools (Plotly, Tableau, Power BI, D3.js).  
- Integrate filters, sliders, and region selectors for user interactivity.  
- Refine visualization aesthetics and user experience (consistent color schemes, clear labels).  
- Highlight notable findings and insights from the interactive dashboard.  

**Evaluation:**
- **Deliverable:** A near-complete interactive dashboard prototype with most planned features implemented.  
- **Success Criteria:** All major visualizations are integrated, and advanced interactivity functions as expected. The dashboard demonstrates key insights from the data.  
<img width="825" height="437" alt="image" src="https://github.com/user-attachments/assets/f5f055df-9b88-4568-a175-bc0ba5060cea" />
<img width="836" height="405" alt="image" src="https://github.com/user-attachments/assets/62a598b4-2e51-4424-b0c6-56fa29e2f50a" />
<img width="940" height="427" alt="image" src="https://github.com/user-attachments/assets/72747af4-5974-4c77-8749-0054c30777b0" />
<img width="938" height="435" alt="image" src="https://github.com/user-attachments/assets/f99097d0-ccf0-4865-a2f3-c1e58c023ad6" />

---

### Milestone 4: Finalization, Testing & Reporting (Weeks 6-8)
**Tasks:**
- Conduct comprehensive testing of the dashboard for functionality, data accuracy, and user experience.  
- Summarize regional and global climate trends.  
- Document methodology, visualizations, and insights in a final project report.  
- *(Optional)* Deploy the dashboard as a web application or presentation.  
- *(Optional)* Document considerations for future enhancements.

**Evaluation:**
- **Deliverable:** Fully tested and stable interactive dashboard, and a comprehensive final project report outlining methodology, visualizations, and key insights.  
- **Success Criteria:** The dashboard is robust and bug-free, meeting all defined objectives. The project report clearly articulates findings and the project process.

 **Deployment link:**
-  https://weatherapp-ahsabg4uzaqoptqzyzfpgr.streamlit.app/

---

##  Submission Details
1. Fork this repository to your own GitHub account, naming it DV-Climatescope-[YourName] (replace [YourName] with your actual name)
2. (Optional) Clone your forked repository to your local machine 
3. Make regular, meaningful commits with for every milestone
4. Push your changes to your forked repository or upload your work manually
5. Make sure to update the repo for every milestone 

---
