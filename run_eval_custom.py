import subprocess

def modify_first_line(filename, new_first_line):
	with open(filename, 'r') as file:
		lines = file.readlines()

	if len(lines) > 0:
		lines[0] = new_first_line + '\n'

	with open(filename, 'w') as file:
		file.writelines(lines)

for cnt in range(3, 42):
	modify_first_line('/home/workspace/home/experiments/experiment_2/checkpoint', 'model_checkpoint_path: "ckpt-' + str(cnt) + '"')
	try:
		subprocess.run(['sh', '/home/workspace/home/eval_2.sh'], check=True)
		print("OK")
	except subprocess.CalledProcessError as e:
		print("Err")
