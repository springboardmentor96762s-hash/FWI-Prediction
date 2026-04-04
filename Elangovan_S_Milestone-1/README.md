## 📁 Repository Contents

### **`elangovan_milestone_1.ipynb`**
The main Jupyter Notebook includes:
- Data loading  
- Cleaning and preprocessing  
- Exploratory data analysis  
- Visualizations  

---

## 🛠 Dependencies

To run this notebook, install the following Python libraries:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### **Libraries Used**
- **Pandas** – Data manipulation and cleaning  
- **NumPy** – Numerical computations  
- **Matplotlib & Seaborn** – Visualizations  
- **Jupyter Notebook** – Running `.ipynb` files  

---

## 📑 Data Schema (Inferred)

| Column Name | Description             | Data Type |
| ----------- | ----------------------- | --------- |
| day         | Day of month            | Integer   |
| month       | Month of year           | Integer   |
| year        | Year of observation     | Integer   |
| Temperature | Air temperature (°C)    | Float     |
| RH          | Relative Humidity (%)   | Float     |
| Ws          | Wind Speed              | Float     |
| Rain        | Rainfall (mm)           | Float     |
| FFMC        | Fine Fuel Moisture Code | Float     |
| DMC         | Duff Moisture Code      | Float     |
| DC          | Drought Code            | Float     |
| ISI         | Initial Spread Index    | Float     |
| BUI         | Buildup Index           | Float     |
| FWI         | Fire Weather Index      | Float     |

---

## 📊 Analysis Overview

The notebook follows a complete data analysis workflow:

### **1. Data Loading & Inspection**
- Load dataset  
- Inspect structure, column names, types  
- View initial records  

### **2. Data Cleaning / Preprocessing**
- Handle missing values
  
**we did not find need to do the below things since the data is good**
- Convert data types
- Parse dates
- Fix inconsistent entries  

### **3. Exploratory Data Analysis (EDA)**
The anaysis done are:
- skewness of the dataset
- correlation between variables
- IQR

### **4. Visualization**
Visuals generated include:
- Correlation heatm
- Box plots  
- Histograms
- Pairplot
  
---

