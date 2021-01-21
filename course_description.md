## Overview

We currently live in an era where most computers possess multiple computing units, and where parallelization is key.
In particular, GPGPUs (General Purpose Graphical Processing Units) are built for massive parallism and they have recently risen to prominence as they are now used for many scientific tasks, such as physics or biological simulations, statistical inference ormachine learning.

In this crash course we will focus on CUDA as well as several CUDA-based API, including openMP GPU offloading and python APIs. 
Through concrete examples we will describe the principles at the core of a successful parallelization attempt. 



## Audience

This course is intended for programmers and computationnal biologists who want to take their first steps with GPU programming.

We will assume no previous knowledge of GPU programing, CUDA, or paralellization techniques, but we require that the participant be proficient in at least one language among python or C++.

## Learning objectives

By the end of the course, the participant will be able to :
 * identify good candidates tasks for GPU accelaration
 * understand the structure of a GPU, including memory handling
 * perform some computations on a GPU, using either python or C++
 * manage memory transfers to the GPU for better performances
 * evaluate their GPU code using profiling



## Prerequisite

*Technical:*

 * A laptop with a wifi connection
 * computer should be the same as the one zoom is used on (to help debugging during practicals)
 **register to google colab in advance**

*Knowledge / competencies:*

Participants should be comfortable working in a Linux/UNIX environment and have some basic experience in programming. 
Some knowledge of C/C++, Fortran or python is necessary.

**Note : link to self assessment questionnaire**






# course Notes 

## course outline

9:00

 * Intro slides (only mentions pitfalls and mem management ?) 1.5 h?
 
( 10:30 break 15min )

 * 1st exercice session : 
 	naive matrix multiplication notebook with integrated interactive exs (1h?)

11:30

 * real life examples : 
 		Kmean (0.33 h )

12:00

 -> lunch break ?

13:00

 		generate (1h to 1.5h)

14:00

 * Common pitfalls : branching , asynchronicity (0.75 h)
 
 * Memory management ( 0.5- h ) (3 slides)

 * examples image manipulation notebooks ( 1+ h )

( 16:00 : break 15min )

 * overlapping and monitoring  (using SPH notebook) (0.5 h )


17:00

 * Outro slides (tips and tricks, other libraries, final remarks) ( 0.25h )

 * free-form exercice ( ??? )





## google colab 

 * free
 * disconnect affter 20 min of idleness
 * need account creation
 * easy to connect to . A couple slides disseminated in advance should do the trick.


## other logisitcal aspects

 * dissemination : website
 * discussion : google doc 



