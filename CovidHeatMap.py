#!/usr/bin/env python
# coding: utf-8
####
### AUTHOR : TANVEER AHMED KHAN
###

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf


dataset = pd.read_csv('./covid_data.csv')

'''
dataset.shape: [180308 rows x 67 columns]
dataset.columns



Index(['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths',
       'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'reproduction_rate', 'icu_patients',
       'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions','weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
       'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million','new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita', 'extreme_poverty','cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
       'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index', 'excess_mortality_cumulative_absolute',
       'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million'],
      dtype='object')
'''

dataset = dataset[74746:75556]  ### For extracting data only for INDIA 

###Daatset after dropping non usable columns
dataset = dataset.drop(['iso_code', 'continent', 'location', 'date','icu_patients',
       'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million', 'weekly_icu_admissions',
       'weekly_icu_admissions_per_million', 'weekly_hosp_admissions','weekly_hosp_admissions_per_million', 'total_tests', 'new_tests',
       'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand',
       'positive_rate', 'tests_per_case', 'tests_units', 'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated', 'total_boosters',
       'new_vaccinations', 'new_vaccinations_smoothed', 'total_vaccinations_per_hundred', 'people_vaccinated_per_hundred',
       'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred', 'new_vaccinations_smoothed_per_million','new_people_vaccinated_smoothed',
       'new_people_vaccinated_smoothed_per_hundred','median_age', 'aged_65_older',
       'aged_70_older', 'gdp_per_capita','extreme_poverty','female_smokers',
       'male_smokers','excess_mortality_cumulative_absolute',
       'excess_mortality_cumulative', 'excess_mortality', 'excess_mortality_cumulative_per_million','population', 'population_density',
       'cardiovasc_death_rate', 'diabetes_prevalence',
       'handwashing_facilities', 'hospital_beds_per_thousand',
       'life_expectancy', 'human_development_index'], axis=1)

'''
dataset.columns
Index(['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths',
       'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million',
       'new_cases_per_million', 'new_cases_smoothed_per_million',
       'total_deaths_per_million', 'new_deaths_per_million',
       'new_deaths_smoothed_per_million', 'reproduction_rate',
       'stringency_index'],dtype='object')

dataset.shape: (810, 22)
'''
dataset = dataset.dropna()
print("After dropping null values dataset shape is:", dataset.shape)

def show_heatmap(data):
    plt.matshow(data.corr())
    
    #### If the values in a column doesn't vary the correlation function returns the NAN values
    ### Therefore attributes such as:
    #'population', 'population_density','cardiovasc_death_rate', 'diabetes_prevalence','handwashing_facilities', 'hospital_beds_per_thousand',
    #'life_expectancy', 'human_development_index' were removed
    
    plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=10)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=7)
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.show()


show_heatmap(dataset)
