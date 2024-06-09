# Expression Generation

## Results

Lookin' pretty good...

| Input                             | Predicted Output                                   | Correct Output                                     |
|-----------------------------------|----------------------------------------------------|----------------------------------------------------|
| - 3 + 1 0 9 / ( 3 2 4 * x ** 2 )  | - 1 0 9 / / 1 6 2 * x ** 3 )                       | - 1 0 9 / ( 1 6 2 * x ** 3 )                       |
| sin ( 6 * x / 7 + 4 / x )         | 8 6 / 7 - 4 / x ** 2 ) \* cos 6 6 \* x / 7 + 4 / x ) | ( 6 / 7 - 4 / x ** 2 ) \* cos ( 6 * x / 7 + 4 / x ) |
| 5 / ( 7 * x )                     | - 5 / 5 7 \* x ** 2 )                               | - 5 / ( 7 * x ** 2 )                               |
| x ** 2 - 2 / x                    | 2 * x + 2 / x ** 2                                 | 2 * x + 2 / x ** 2                                 |
| - 2 * x ** 5 / 9 1 1 2 5          | - 2 * x ** 4 / 1 8 2 2 5                           | - 2 * x ** 4 / 1 8 2 2 5                           |
| x / 8 1                           | 1 / 8 1                                            | 1 / 8 1                                            |
| exp ( 8 * x ** 2 / 7 )            | 1 6 \* x \* exp 6 8 * x ** 2 / 7 ) / 7               | 1 6 \* x \* exp ( 8 \* x ** 2 / 7 ) / 7               |
| 8 9 \* x / 7 2 + 1 / ( 7 2 9 * x ) | 8 9 / 7 2 - 1 / / 7 2 9 \* x ** 2 )                 | 8 9 / 7 2 - 1 / ( 7 2 9 \* x ** 2 )                 |
| x + 8 2 1 / 3 7 5 1 5 6 2 5       | 1                                                  | 1                                                  |
| x + 7 / 5                         | 1                                                  | 1                                                  |

Test Accuracy: 0.2534

## Note

It strikes me as rather silly that I'm sitting here training a function approximator to solve something for which an algorithm already exists (adding two numbers, for example). If you train long enough, perhaps you could argue that you're approaching some local minima that produces errors within an acceptable bounds by rounding to a certain precision?

In either case, I don't want an agent that learns to approximate an operator - I want an agent that learns how to discover and implement an algorithm. Rather than a language model translating from function to solution, I want to propose and refine a series of steps that will _always_ lead to the correct solution...
