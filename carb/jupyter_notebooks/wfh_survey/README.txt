********************************************************************************************
README - Code and data for "Why Working from Home will Stick", by Barrero, Bloom, and Davis

- Jose Maria Barrero (jose.barrero@itam.mx www.jmbarrero.com), April 2022
********************************************************************************************

***********
* Overview
***********

This package contains data and Stata code files to:

	(1) reproduce results from the 28 April 2021 version of our paper on "Why Working From Home Will Stick" (the same as NBER WP version that came out on 3 May 2021). These results use data from the May 2020 to March 2021 waves of the Survey of Working Arrangements and Attitudes (SWAA).

	(2) produce updated results from each of the figures and tables in the 28 April 2021 version of the paper, but also including data from April 2021 and later survey waves.

We also include a Creative Commons License CC-BY 4.0 license https://creativecommons.org/licenses/by/4.0/ associated with this work.


***********
* Citation
***********


When using these data please cite:

	Barrero, Jose Maria, Nicholas Bloom, and Steven J. Davis, 2021. “Why working from home will stick,” National Bureau of Economic Research Working Paper 28731.


If you wish, you may also cite the data separately from the paper above.


*****************
* Included files
*****************

The package includes the following files:
	
	- WFHdata_March22.csv: contains anonymous microdata from all waves of the Survey of Working Arrangements and Attitudes (SWAA) collected since May 2020. This version of the file was created in Apri 2022. Each row is a survey response.
	
	- CPSdata.csv: contains data from a pooled sample of 2010 to 2019 CPS responses, which we use to assess how representative the SWAA data are. The underlying CPS data were extracted in June 2020. Each row corresponds to an observation from the CPS.

	- Variable dictionary Apr22.pdf: This PDF file contains a description of all variables in WFHdata_March22.csv. These descriptions are then coded as variable labels in the Stata code that replicates the results (see WFH_WPresults_Master_March22.do and WFH_WPresults_Master_March22.do, below). The PDF also contains the descriptions of categorical variables. E.g., a variable on current working status takes a value of 1 if the respondent is "Working on business premises," 2 if "Working from home," or 3 if "Not working." The PDF details these categorical descriptions for all of the relevant variables, and the code file formally labels the values within Stata. The variable dictionary also flags variables that could have substantial measurement error but may be of use to other researchers, in particular "occupation."
	
	- WFH_WPresults_Master_March22.do: This file runs all of the necessary code that in-files the data from the two CSV files into Stata and reproduces all the figures and tables in the 28 April 2021 working paper. We also include lines of code that reproduce most of the estimates reported in the paper, but that are not part of the tables and figures.


	- WFH_updatedresults_Master_March22.do: This file runs all of the necessary code that in-files the data from the two CSV files into Stata and produces UPDATED VERSIONS all the figures and tables in the paper. These results won't exactly match the 28 April 2021 working paper because they also include data from survey waves after March 2021. We also include lines of code that replicate most of the estimates reported in the paper, but that are not part of the tables and figures.


Note: both of the above .do files use the same source data and output nearly identical versions of the processed .dta file called "WFHdata_fromCSV.dta". They then use slightly different code to generate updated results or reproduce results from the 28 April 2021 working paper version, respectively. For November 2021 and later releases, the .dta output from "WFH_updatedresults_Master_December21.do" modifies the variable `wfh_eff_COVID_quant' to fix a coding error. "WFH_WPresults_Master_December21.do" does not make that update to preserve the exact reproducibility of the 28 April 2021 NBER WP results.

	- LicenseFebruary2022.txt: Includes the CC-BY 4.0 license associated with this work.

	
******************************************************************************
* Additional notes on the target population and how we construct the weights. 
******************************************************************************

A note on the target population and on how we re-weight the raw SWAA data:
	
	The 28 April 2021 version of our working paper uses a target population of US residents ages 20 to 64 who earned at least $20,000 in 2019. We pool all responses collected from a given survey provider (see the working paper for details) across the May 2020 to March 2021 survey waves and re-weight them to match the share of people in a given {age x sex x education x earnings} cell in the 2010-2019 CPS. Observation i in the pooled sample gets weighting factor 

	w = ( raw share of observations in i's {age x sex x education x earnings} cell in the pooled sample) /  ( share of observations in i's {age x sex x education x earnings} cell in the 2010 -2019 CPS ). 

We then rescale the resulting weighting factor variable (`cratio100_2021m3') so it adds up to 100. This weighting factor should be used to reproduce the exact results from the working paper, as done in the included file called WFH_WPresults_Master_May21.do.


	For April 2021 and later waves of the SWAA, we construct sample weights as follows. Observation i in month t gets weighting factor 
	
	w = ( raw share of observations in i's {age x sex x education x earnings} cell in the 6 months comprising t-5 to t in the SWAA) /  ( share of observations in i's {age x sex x education x earnings} cell in the 2010 -2019 CPS ). 

The numerator uses 6 months of SWAA data to prevent the weighting factor for a given cell from varying significantly across survey waves. We then rescale the resulting weighting factor variable (`cratio100') so it adds up to 100 across ALL waves of the SWAA.

	Starting with the April 2021 wave of the SWAA we expanded the target population to also include persons who earned $10,000 to $20,000 in 2019. In accordance with the 6-month smoothing of the weights (see previous paragraph), we are gradually phasing in this earnings group into the sample over 6 months. In April 2021, this group's sample weights target 1/6 of their proportion in the 2010-2019 CPS. In May 2021, they target 2/6, and so on. In September 2021, the new earnings group should add up to its full weight in the corresponding CPS sample.

	Starting with the July 5, 2021 data release we have included a version of the weights that winsorize the weights such that the relative weight of any given observation does not exceed 5/N where N is the total (raw) number of observations. 

	Starting with the October 5, 2021 data release, we use these winzorized weights to generate all updated results. The variable `cratio100' contains the winsorized weights in the October 5, 2021 and later data releases and `cratio100_nw' contains the raw weights. In prior releases `cratio100' contains the raw weights and `cratio100_w' contains the winterized weights (when available).

	

*************************
* How to use the weights
*************************

For most applications, we recommend using the provided weights, even when considering only subsamples of the data. Namely, for the full sample that includes all survey waves use `cratio100', and for May 2020 to March 2021 only (the sample in our May 3, 2021 NBER WP) use `cratio100_2021m3'.

	We recommend this approach because our weights are constructed by pooling several months of data together (see above), and so avoid up- or down-weighting particular cells by large amounts when they show up rarely in the raw survey data. Thus, if you want to construct month-by-month estimates of the share of full paid days worked from home, for example, we recommend you weight by `cratio100' or `cratio100_2021m3' rather than constructing month-specific weights.


**************************************
* Release notes for the data and code
**************************************

	*********************************************************************
	* August 5, 2021 version covering May 2020 to July 2021 survey waves

	- Starting with this version, the code that generates an updated version of Figure 1 from the NBER WP paper uses a global macro to label the time axis in that figure.

	- We fixed a bug in the code that generates an updated version of Figure 3.

	
	*************************************************************************
	* September 5, 2021 version covering May 2020 to August 2021 survey waves

	- We fixed a bug that made the cratio100 weights add up to nearly but not exactly 100.

	- Starting with this version of the code, we fix a bug that affected the variable "wfh_eff_COVID_quant" in the "WFH_updatedresults_Master_August21.do" file that produces updated results.

	- We include code that produces additional results not in the 28 April 2021 NBER working paper version at the end of "WFH_updatedresults_Master_August21.do". Many of these results are posted on the home page of wfhresearch.com shortly after the data release.


	**************************************************************************
	* October 5, 2021 version covering May 2020 to September 2021 survey waves

	- Starting with this data release, `cratio100' contains sample weights that are winsorized such that the relative weight of any given observation does not exceed 5/N where N is the total (raw) number of observations. The variable `cratio100_nw' contains the raw weights. In prior months `cratio100' contained raw weights and `cratio100_w' contained winterized weights.

	- We added a variable on self employment.

	- We added code to reproduce a few new results not in the 28 April 2021 NBER Working Paper.

	
	*************************************************************************
	* November 5, 2021 version covering May 2020 to October 2021 survey waves

	- No methodological updates

	- Added new variables on quits, anxiety related to being on business premises, and coordination on WFH versus on-site days. See variable dictionary


	**************************************************************************
	* December 5, 2021 version covering May 2020 to November 2021 survey waves

	- No methodological updates

	- Added new variables related to commuting: time required to groom/prepare for work while WFH versus on premises, commuting modes pre-COVID and current, time leaving for work pre-COVID and currently, the distribution of work time across locations (current, desired post-COVID, and planned post-COVID), whether the worker has the option to work at more than one worksite, and on the ability to work from home. See variable dictionary

	*************************************************************************
	* January 5, 2022 version covering May 2020 to December 2021 survey waves

	- We began a transition to modify the language in several questions, including the core questions about worker desired and employer planned post-COVID work from home.
		-  The legacy language says "After COVID, in 2022 and later," and the new language says "After the pandemic ends." 
		- In the December 2021 wave we randomly assigned respondents to the legacy or the new versions of the relevant questions (50/50 each). Each respondent saw consistent language across the affected questions.
		- Starting with January 2022 we plan to use only the new language for all respondents.
		- We found only modest differences in the response patterns across the new and old language.

	- We included a new version of the question on who decides what days are work-from-home days and what days are for work on premises (`who_decides_wfhdays'). Note that the question language is slightly different from when we previously asked the same question.

	- We changed the question about why workers cannot be 100% remote (worktime_nonremotable_why) to allow for multiple reasons. Thus, starting with the December 2021 release there are binary variables for each of the choices, rather than a single categorical variable for the primary reason.

	- We added a new variables about the respondent's working status in each day of the reference week ("last week"), namely Monday, Tuesday,...,Sunday. See `workstatus_monday' and similar variables for other days of the week. Using these days-specific workstatus variables, we construct a new measure of the current amount of working from home contained in the variable `wfhcovid_fracmat.' We first implemented this new approach and calculate this measure starting with the November 2021 survey wave.

	- We also added new variables about work-from-anywhere, preferences about the choice of when to come into business premises (i.e. on what days), and two new attention check variables (`cities_attn' and `grass_color_attnfull').


	*************************************************************************
	* February 5, 2022 version covering May 2020 to January 2022 survey waves

	- We implement a change in methodology for the variable `numwfh_days_postCOVID_boss_s_u' containing employer plans for post-COVID working from home (as reported by the respondent). 
		- In the May 2020 thru December 2021 waves (inclusive) of the SWAA we impute ZERO post-COVID working from home for respondents who say "My employer has not discussed this matter with me or announced a policy about it" in response to the relevant survey question. 
		- Starting with the January 2022 survey wave, we have stopped projecting ZERO WFH days for respondents that report no clear employer plans for post-COVID working from home AND are currently doing at least some WFH. We assume these workers will do as much WFH post-COVID as the average worker who is currently doing at least some WFH AND reported employer plans.
		- The February 2022 and later releases include a variable `numwfh_days_postCOVID_boss_s_u_l' that preserves the old methodology.

	- We added new variables about personal grooming habits while working from home versus coming into work, working from home schedules, commuting to "third" locations for remote work, and party affiliation.
 
	- We added code at the very end of the WFH_updatedresults_Master_January22.do file that reproduces some new results from the survey, for example results about personal grooming habits while working from home versus on business premises.
		
	*************************************************************************
	* March 5, 2022 version covering May 2020 to February 2022 survey waves

	- We modified the definition of the variable `wfhcovid_fracmat' so that it is missing for workers who reported they did not work in the reference week. The question on which we base this variable (asking about working status for each specific day of the reference week) is seen by all respondents. `wfhcovid_fracmat_all' includes data from all respondents, including those who did not report working in the reference week.

	- We added code that reproduces some of the new results in the March 2021 updates pack posted on wfhresearch.com, specifically about: (1) the attitudes and preferences of persons who are unemployed and searching for a job and persons who are not working and seeking work, (2) benefits of working from home versus the benefits of working on business premises.

	*************************************************************************
	* April 5, 2022 version covering May 2020 to March 2022 survey waves

	- We implement a change regarding the main labor earnings variable associated with a given respondent. Up until December 2021, we focused on 2019 earnings. We avoided using 2020 earnings, which could be substantially disrupted by the pandemic recession.
		
		- In the January and February 2022 survey waves, we collected data on labor earnings for both 2019 and 2021 and examined data from all respondents, whether they met the $10,000 earnings cutoff in either year or not. For the February 5 and March 5, 2022 data releases we manually imposed the $10,000 cutoff using 2019 earnings, maintaining our historical sample selection criterion. (We collected a larger sample on N = 7,500 in January and February 2022 so that we could compare 2019 and 2020 earnings regardless of the cutoff and then implement the cutoff manually before analyzing and releasing the new data.)

		- Examining the data for 2019 and 2021 earnings side by side, we found that meeting the $10,000 cutoff in one year predicted meeting the cutoff in the other year with high probability (85% or more).

		- The April 5, 2022 release implements a transition to using 2021 as the reference year for earnings. We implement this transition retroactively for data collected between January and March 2022. Namely, we use randomization to determine the reference year for a respondents earning's. 
			- In January 2022, 25% of respondents have 2021 as the reference year and 75% have 2019.
			- In February 2022, 50% of respondents have 2021 as the reference year and 50% have 2019.
			- In March 2022, 75% of respondents have 2021 as the reference year and 25% have 2019.
			- We plan to phase out the survey question on 2019 earnings entirely in April 2022, so 100% of respondents will have 2021 as the reference year in April and subsequent months.

		- We include a new variable `earnings_year' showing the reference year for earnings (itself coded categorically in `income_cat' and `iincomebin', and continuously in `income' and `logincome') for each respondent. The reference year is 2019 for all responses collected in 2020 in 2021, and depends on the randomization across 2019 and 2021 for data collected in the first 3 months of 2022.

		- Due to this methodological transition, there will be small discrepancies between the January and February 2022 data as released on February 5 and March 5, 2022 and the data for those months released on April 5, 2022. We have not found these discrepancies to materially change any results we have published previously using those earlier data releases.

	- We intend to continue using prior-year earnings as we continue our survey collection efforts into the future. So, in early 2023 we anticipate implementing a similar transition to using 2022 as the reference year for earnings.
	

