# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Was 2015 anomalous?
# 
# 1990 and 2015 featured the most no-hitters of any season of baseball (there were seven). Given that there are on average 251/115 no-hitters per season, what is the probability of having seven or more in a season?
# - Draw 10000 samples from a Poisson distribution with a mean of 251/115 and assign to n_nohitters.
# - Determine how many of your samples had a result greater than or equal to 7 and assign to n_large.
# - Compute the probability, p_large, of having 7 or more no-hitters by dividing n_large by the total number of samples (10000).
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(2.1826086956521737,size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large/10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)

# %% [markdown]
# The result is about 0.007. This means that it is not that improbable to see a 7-or-more no-hitter season in a century. We have seen two in a century and a half, so it is not unreasonable.
# %% [markdown]
# # PDF - Função Densidade de Probabilidade(FDP)
# Em teoria das probabilidades e estatística, a função densidade de probabilidade (FDP), ou densidade de uma variável aleatória contínua, é uma função que descreve a verossimilhança de uma variável aleatória tomar um valor dado. A probabilidade da variável aleatória cair em uma faixa particular é dada pela integral da densidade dessa variável sobre tal faixa - isto é, é dada pela área abaixo da função densidade mas acima do eixo horizontal e entre o menor e o maior valor dessa faixa. A função densidade de probabilidade é não negativa sempre, e sua integral sobre todo o espaço é igual a um. A função densidade pode ser obtida a partir da função distribuição acumulada a partir da operação de derivação (quando esta é derivável).
# %% [markdown]
# # CDF Função Distribuição Cumulativa(FDC)
# Em teoria da probabilidade, a função distribuição acumulada (fda) ou simplesmente função distribuição, descreve completamente a distribuição da probabilidade de uma variável aleatória de valor real X. 

# %%
def ecdf(data):
    l = len(data)
    return np.sort(data), np.arange(1, l + 1) / l

# %% [markdown]
# # The Normal PDF
# In this exercise, you will explore the Normal PDF and also learn a way to plot a PDF of a known distribution using hacker statistics. Specifically, you will plot a Normal PDF for various values of the variance.

# %%

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, 100000)
samples_std3 = np.random.normal(20, 3, 100000)
samples_std10 = np.random.normal(20, 10, 100000)

# Make histograms
plt.hist(samples_std1, bins=100, density=True, histtype='step')
plt.hist(samples_std3, bins=100, density=True, histtype='step')
plt.hist(samples_std10, bins=100, density=True, histtype='step')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

# %% [markdown]
# # The Normal CDF
# Now that you have a feel for how the Normal PDF looks, let's consider its CDF. Using the samples you generated in the last exercise (in your namespace as samples_std1, samples_std3, and samples_std10), generate and plot the CDFs.
# - Use your ecdf() function to generate x and y values for CDFs: x_std1, y_std1, x_std3, y_std3 and x_std10, y_std10, respectively.
# - Plot all three CDFs as dots (do not forget the marker and linestyle keyword arguments!).
# - Hit submit to make a legend, showing which standard deviations you used, and to show your plot. There is no need to label the axes because we have not defined what is being described by the Normal distribution; we are just looking at shapes of CDFs.

# %%
# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)
# Plot CDFs
plt.plot(x_std1, y_std1, marker='.',linestyle='none')
plt.plot(x_std3, y_std3, marker='.',linestyle='none')
plt.plot(x_std10, y_std10, marker='.',linestyle='none')


# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()

# %% [markdown]
# # Are the Belmont Stakes results Normally distributed?
# Since 1926, the Belmont Stakes is a 1.5 mile-long race of 3-year old thoroughbred horses. Secretariat ran the fastest Belmont Stakes in history in 1973. While that was the fastest year, 1970 was the slowest because of unusually wet and sloppy conditions. With these two outliers removed from the data set, compute the mean and standard deviation of the Belmont winners' times. Sample out of a Normal distribution with this mean and standard deviation using the np.random.normal() function and plot a CDF. Overlay the ECDF from the winning Belmont times. Are these close to Normally distributed?
# 
# Note: Justin scraped the data concerning the Belmont Stakes from the Belmont Wikipedia page.
# 1. Compute mean and standard deviation of Belmont winners' times with the two outliers removed. The NumPy array belmont_no_outliers has these data.
# 2. Take 10,000 samples out of a normal distribution with this mean and standard deviation using np.random.normal().
# 3. Compute the CDF of the theoretical samples and the ECDF of the Belmont winners' data, assigning the results to x_theor, y_theor and x, y, respectively.
# 4. Hit submit to plot the CDF of your samples with the ECDF, label your axes and show the plot.

# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
file_path = os.getcwd()
belmont = pd.read_csv (file_path+'\data\\belmont.csv', delimiter=',')
belmont['Time'] = pd.to_timedelta(belmont['Time'].map('00:0{}'.format).replace('.',':')).dt.total_seconds()


# %%
belmont_no_outliers = belmont.drop(belmont[(belmont['Year'] == 1970) | (belmont['Year'] == 1973)].index, axis=0)

# %% [markdown]
# ## Compute mean and standard deviation: mu, sigma

# %%
mu = belmont_no_outliers['Time'].mean()
sigma = belmont_no_outliers['Time'].std()


# %%
# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers['Time']) 


# %%
# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()

# %% [markdown]
# -- The theoretical CDF and the ECDF of the data suggest that the winning Belmont times are, indeed, Normally distributed. This also suggests that in the last 100 years or so, there have not been major technological or training advances that have significantly affected the speed at which horses can run this race.


# %% [markdown]
# ## What are the chances of a horse matching or beating Secretariat's record?
# Assume that the Belmont winners' times are Normally distributed (with the 1970 and 1973 years removed), what is the probability that the winner of a given Belmont Stakes will run it as fast or faster than Secretariat?


# %%
# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <=144)/1000000

# Print the result
print('Probability of besting Secretariat:', prob)


# %% [markdown]
# ## If you have a story, you can simulate it!
# Sometimes, the story describing our probability distribution does not have a named distribution to go along with it. In these cases, fear not! You can always simulate it. We'll do that in this and the next exercise.
# 
# In earlier exercises, we looked at the rare event of no-hitters in Major League Baseball. Hitting the cycle is another rare baseball event. When a batter hits the cycle, he gets all four kinds of hits, a single, double, triple, and home run, in a single game. Like no-hitters, this can be modeled as a Poisson process, so the time between hits of the cycle are also Exponentially distributed.
# 
# How long must we wait to see both a no-hitter and then a batter hit the cycle? The idea is that we have to wait some time for the no-hitter, and then after the no-hitter, we have to wait for hitting the cycle. Stated another way, what is the total waiting time for the arrival of two different Poisson processes? The total waiting time is the time waited for the no-hitter, plus the time waited for the hitting the cycle.
# 
# Now, you will write a function to sample out of the distribution described by this story.


# %%
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size=size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size=size)

    return t1 + t2


# %% [markdown]
# ## Distribution of no-hitters and cycles
# Now, you'll use your sampling function to compute the waiting time to observe a no-hitter and hitting of the cycle. The mean waiting time for a no-hitter is 764 games, and the mean waiting time for hitting the cycle is 715 games.

# %%
# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764,715,100000)

# Make the histogram
_ = plt.hist(waiting_times, bins=100, density=True, histtype='step')


# Label axes
_ = plt.xlabel('times')
_ = plt.ylabel('Percents')


# Show the plot
_ = plt.show()

# %% [markdown]
# Great work! Notice that the PDF is peaked, unlike the waiting time for a single Poisson process. For fun (and enlightenment), I encourage you to also plot the CDF.
# %%
