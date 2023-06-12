## SECTION 1 : PROJECT TITLE

## **Home BUDDY**

![Image text](/resources/home_page.png)

------

## SECTION 2 : EXECUTIVE SUMMARY

Singapore is well-known as a hub for economy and education. As such, there has been many foreigners and expatriates that come for various purposes. A common struggle faced by foreigners is the issue of finding an accommodation for their long-term stay in Singapore. While it’s true that websites such as PropertyGuru has a wide range of   vailable filters that a user can apply, there are many crucial information in the description that can’t be filtered using the built-in hard filters. For example:

i)      Landlords looking for specific demographics of tenants (e.g ethnicity, gender, profession, age groups, marital status, etc)
ii)     Landlords with specific requests (e.g. no cooking, no smoking, comfortable with pets, etc)
iii)    Hidden costs (e.g. rentals not including utility bills, agent fees, etc)
iv)     Amenities provided (e.g. air-conditioning, cooking, etc)
v)      Presence of landlord in the house or number of household members in the house.

Currently, these information are usually stated only in the description and there are no quick ways to filter the descriptions except to click into the link and read. This causes the whole process is very time consuming and can take hours on end, especially if the postings that were clicked on constantly fail to meet the user’s pre-requisites.

As such, our project applied Natural Language Processing (NLP) methods to speed up this process and dynamically meet the user’s needs to be more relevant than hard-filtering. This project is not limited by constraints set by hard-coded values as users can most likely write anything they want. Combined with RPA, our system performs real-time scraping and analysis of description to generate the closest shortlist given the user’s constraints for the user to have a quick overview for decision-making. And we use a chatbot to collect the user preferences.

------

## SECTION 3 : CREDITS

| Official Full Name      | Student ID (MTech Applicable) | Email (Optional)   |
| ----------------------  | ----------------------------- | ------------------ |
| Pradeep Kumar Arumugam  | A0261606J                     | pradeepkumaravsp@gmail.com |
| Zhao Lutong             | A0249279L                     |                    |
| Jonathan Lim Ching Loong| A0261707E                     |                    |


------

## SECTION 4 : USER GUIDE

[UserGuide](/User%20Guide/USER%20GUIDE.pdf)

### [ 1 ] To run the webapp in local machine

>  (Notes: Preferred Python 3.7 or above)

> Open up your command prompt/terminal

> Do git clone this repository or go to git url mentioned above to directly download the project, unzip and keep it in your folder directory.

> pip install the components from requirements.txt

> cd <your folder path>\HomeBuddy

> Run the server.py with python
>
> - py server.py
> - py3 server.py (if you have different versions of python)

> Once it’s done, you’ll see this line in your command prompt “Running on http://192.168.10.106:8080/ (Press CTRL+C to quit)” wither copy paste this url or directly go to any of your web browser and type http://localhost:8080/ and you’ll be able to access our webapp



### [ 2 ] To run the webapp in VM

> Open up your terminal

> git clone <web url to this repository>

> pip install the components from RentalBuddy\requirements.txt

> cd <your folder path>\RentalBuddy	

> Run the server.py with python
>
> - $python server.py
>
> - $python3 server.py (if you have different versions of python)

> Go to URL using web browser http://localhost:8080/ or http://192.168.10.106:8080/


Datasource we used:

​[Property Guru](https://www.propertyguru.com.sg/)

------

## SECTION 5 : PROJECT REPORT / PAPER

[ProjectReport](/Project%20Report/Practice%20Module%20Report_Group%2010.pdf)

------

## SECTION 6 : MISCELLANEOUS


