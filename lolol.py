import pandas as pd
import numpy as np
from datetime import datetime, timedelta

date_today = datetime.now()
days = pd.date_range(date_today, date_today + timedelta(365), freq='B').strftime('%Y-%m-%d')
df = pd.DataFrame({'open': np.nan}, index=days)

print(df)