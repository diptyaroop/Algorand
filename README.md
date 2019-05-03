# Algorand Simulator

CS 620 course project

Members:
Diptyaroop Maji (183050016)
Debayan Bandyopadhyay (183050034)
Abhishek Varma (183050038)

Built a (somewhat) working Algorand discrete event simulator on Python (using heapq module).

Algorand block generation for normal cases --> done
(Tested with 256 nodes, 50 blocks generated)

Algorand block generation with fail stop adversary --> done
(Adversary does not propose block even if proposer, and does not take part in consensus)

Algorand with Byzantine adversary --> 20% done (till adversary proposing 2 blocks at once)
