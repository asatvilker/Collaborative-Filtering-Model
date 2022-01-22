**Collaborative filetering model (K-nearest neighbours)**

* Using a k-nearest neighbours approach with mean values to adjust weighting
* Incorporated a basic GUI to demonstrate the model in action
* The model looks at user ratings for a variety of different restaurants and then computes the recommendations based on what similar users liked

**What I took from this project:

This was my first attempt at building a recommender system and my first attempt implementing collaborative filtering techniques. I would say my assumption that these models must be very hard to implement was wrong. The many libraries out there take care of it. The real complexity is understanding your data and the use case, tuning the hyper-parameters. E.g. do we care about the predicted rating or is retrieval (top-K) more important. It also highlighted the importance in future of using data preprocessing techniques before training your data. Being new to this type of work I had not thought too much about the quality of the data, just the quantity however now I see it as the opposite.

**Looking forward:

Next I am keen to explore deep learning based solutions to recommender systems, something I will be focusing on in my third year project. I expect it to be more challenging, from both a theoretical and implementational perspective!

**Setup:**
Run the following to ensure you have all the required dependencies
    
```pip install numpy```

```pip install pandas```

```pip install scikit-surprise```

Thank you for reading!
