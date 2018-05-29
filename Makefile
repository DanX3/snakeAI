CC=python2.7
PROFILEOUT=profile_data
PROFILELINES=profile_lines
FLAME=profile.flame
FLAMEIMG=profile.svg
FLAMEOPTS= --title "Snake AI" --colors blue

all: run

run:
	$(CC) SnakeAI.py

flame:
	python -m vmprof -o $(PROFILEOUT) SnakeAI.py
	vmprof-flamegraph.py $(PROFILEOUT) > $(FLAME)
	flamegraph.pl $(FLAMEOPTS) $(FLAME) > $(FLAMEIMG)
	firefox $(FLAMEIMG)

clean:
	rm neat-checkpoint* .*.swp
