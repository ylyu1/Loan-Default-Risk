from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count, sum as fsum, udf, regexp_replace, length, avg, broadcast
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

# Start a Spark session
spark = SparkSession.builder.appName("LoanDataAnalysis").getOrCreate()

# Load the dataset
df = spark.read.csv('data.csv', header=True, inferSchema=True)

# Drop unnecessary columns
df = df.drop('id', 'issue_d', 'installment', 'last_credit_pull_d', 'last_fico_range_high', 'last_fico_range_low')

# Handling duplicates
df = df.dropDuplicates()

# Preprocessing: Missing value flags and dropping columns with too many missing values
columns_with_missing = [c for c in df.columns if df.filter(df[c].isNull()).count() > 0]

def missing_distribution_by_loan_status(df, column):
    name = column + '_missing'
    df = df.withColumn(name, when(col(column).isNull(), 1).otherwise(0))
    grouped_counts = df.groupBy('loan_status', name).count()
    counts_unstacked = grouped_counts.groupBy('loan_status').pivot(name).sum('count')
    counts_unstacked = counts_unstacked.withColumn('missing_percentage', (col('1') / (col('1') + col('0'))) * 100)
    counts_unstacked.show()

for column_name in columns_with_missing:
    missing_distribution_by_loan_status(df, column_name)

df = df.drop('mths_since_last_record', 'mths_since_last_delinq', 'inq_last_12m', 'emp_length')

# Convert term to integer after stripping "months"
df = df.withColumn('term', regexp_replace('term', 'months', '').cast('int'))

# Handling categorical columns and binary encoding
df = df.withColumn('verification_status_binary', when(col('verification_status') == 'Not Verified', 0).otherwise(1))
df = df.drop('verification_status')

# Feature engineering for 'purpose' column
top_categories = ['debt_consolidation', 'credit_card', 'home_improvement', 'other']
df = df.withColumn('purpose', when(col('purpose').isin(top_categories), col('purpose')).otherwise('other'))

# Create new features from 'earliest_cr_line'
@udf(returnType=IntegerType())
def extract_year(date_str):
    return int(date_str[-4:])

df = df.withColumn('earliest_cr_line_year', extract_year(col('earliest_cr_line')))
df = df.withColumn('cr_history_to_2015', 2015 - col('earliest_cr_line_year'))
df = df.drop('earliest_cr_line')

# Removing outliers in DTI
df = df.filter(col('dti') <= 100)

# New combined features
df = df.withColumn('loan_annual_income_ratio', col('loan_amnt') / col('annual_inc'))
df = df.withColumn('delinquency_now_plus_past_2yrs', col('acc_now_delinq') + col('delinq_2yrs'))
df = df.withColumn('delinq_amnt_per_acc', col('delinq_amnt') / (1 + col('acc_now_delinq')))
df = df.withColumn('fico_per_inquiries', col('avg_fico_score') / (1 + col('inq_last_6mths')))

# Feature scaling using MinMaxScaler
features_to_scale = ['loan_amnt', 'annual_inc', 'dti', 'loan_annual_income_ratio', 'delinquency_now_plus_past_2yrs',
                     'delinq_amnt_per_acc', 'fico_per_inquiries', 'cr_history_to_2015']
assembler = VectorAssembler(inputCols=features_to_scale, outputCol="features")
scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")

pipeline = Pipeline(stages=[assembler, scaler])
model = pipeline.fit(df)
scaled_df = model.transform(df)

# Select the final set of columns for the model
final_df = scaled_df.select('scaled_features', 'loan_status')

# Save or show the final DataFrame
final_df.show()

# Stop the Spark session
spark.stop()
