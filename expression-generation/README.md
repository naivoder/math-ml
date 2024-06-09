# Expression Generation

It strikes me as rather silly that I'm sitting here training a function approximator to solve something for which an algorithm already exists (adding two numbers, for example). If you train long enough, perhaps you could argue that you're approaching some local minima that produces errors within an acceptable bounds by rounding to a certain precision?

In either case, I don't want an agent that learns to approximate an operator - I want an agent that learns how to discover and implement an algorithm. Rather than a language model translating from function to solution, I want to propose and refine a series of steps that will _always_ lead to the correct solution...
