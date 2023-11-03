run:
	cp tetris_dx.sgb.ram backup.ram
	./game.py
	cp backup.ram tetris_dx.sgb.ram 
