# ****SOF output file analysis****

General information about the files in the `SOF-OFA` folder:

[Go to Folder SOF-DATA](https://drive.google.com/drive/folders/1IhZcCoqHQuE_E8T6JtG1WAsSqN-iB9Ss?usp=sharing)

### Consider the following data (located in the specified folder):
```bash
    1- The `Answer` table, which includes the following information from the answers published on `SOF`:
```

* **Answer number** .<br>
* **Date of the answer** .<br>
* **Number of votes for the answer** .<br>
* **Responder's ID** .<br>
* **Number of comments** .<br>
* **Closure date (only not null if the answer has been closed)** .<br>

```bash
    2- The 'Question' table, which contains the following information from the questions published on 'SOF':
```
* **Question number** .<br>
* **Date of the question** .<br>
* **Number of votes for the question** .<br>
* **Number of times the question was viewed** .<br>
* **Asker's ID** .<br>
* **Number of comments** .<br>
* **Closure date (only not null if the question has been closed)** .<br>
* **Number of times the question was marked as 'Favorite'** .<br>

```bash
    3- The 'Q-A' table, which includes the following information from the questions and answers published on 'SOF':
```
* **Question number** .<br>
* **Answer number** .<br>
* **Is the answer marked as 'Accepted'** .<br>

```bash
    4- The 'U' table includes the following information from active individuals on 'SOF' (askers or responders):
```
* **Person's ID** .<br>
* **Person's 'Reputation'** .<br>
* **Number of times the person's page was viewed** .<br>
* **Number of 'UPvotes'** .<br>
* **Number of 'Downvotes'** .<br>

```bash
    5- The 'user-badge' table, which contains the following information about individuals:
```
*  **Person's ID** .<br>
*  **Name of the 'badge' received by the person"** .<br>
<br>


## ***Using data visualization, you accept or reject the following hypotheses:***

**1**- Questions with an `Accepted` answer receive more `Views`.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q1)


**2**- Questions with more `Views` receive more `Comments`.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q2)


**3**- An answer receives more likes if the related question has many likes.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q3)


**4**- Individuals with high `Reputation` tend to respond to questions that receive many likes.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q4)


**5**- Individuals with high `Reputation` tend to respond to closed questions.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q5)


**6**- Individuals with a higher number of `Badges` have higher `Reputation`.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q6)


**7**- The shorter the time delay in responding (the less time between the question and the answer), the higher the chance of the answer being accepted.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q7)


**8**- If we categorize questions into four groups: unanswered, with few answers, with a moderate number of answers, and with many answers, the likelihood of getting likes increases from right to left.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q8)


**9**- The number of `Views` a question receives correlates with the number of likes it gets.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q9)


**10**- If individuals are divided into three categories of `Low-Medium-High Reputation`, the questions from individuals with `High Reputation` receive more likes.
[Go to Answar](https://github.com/Hadikamali/SOF-output-file-analysis-part-two/tree/main/Answer-Q10)


----------

