import os

jobName = 'Demo_Train_for_lowfps'
part = 'Pixel'  # node part

# name of computational nodes which are available
freeNodes = ['SH-IDC1-10-5-30-135', 'SH-IDC1-10-5-30-138']


ntaskPerNode = 8  # number of GPUs per nodes
reuseGPU = 0
envDistributed = 1
# gpu id on computational node
gpuDict = "\"{\'SH-IDC1-10-5-30-135\': \'0,1,2,3,4,5,6,7\', \'SH-IDC1-10-5-30-138\': \'0,1,2,3,4,5,6,7\'}\""

nodeNum = len(freeNodes)
nTasks = ntaskPerNode * nodeNum if envDistributed else 1
nodeList = ','.join(freeNodes)
initNode = freeNodes[0]

scrip = 'train'
config = 'configEVI'  # config name (configEVI.py here)


def runDist():
    pyCode = []
    pyCode.append('python')
    pyCode.append('-m')
    pyCode.append(scrip)
    pyCode.append('--initNode {}'.format(initNode))
    pyCode.append('--config {}'.format(config))
    pyCode.append('--gpuList {}'.format(gpuDict))
    pyCode.append('--reuseGPU {}'.format(reuseGPU))
    pyCode.append('--envDistributed {}'.format(envDistributed))
    pyCode.append('--expName {}'.format(jobName))
    pyCode = ' '.join(pyCode)

    srunCode = []
    srunCode.append('srun')
    srunCode.append('--gres=gpu:{}'.format(ntaskPerNode))
    srunCode.append('--job-name={}'.format(jobName))
    srunCode.append('--partition={}'.format(part))
    srunCode.append('--nodelist={}'.format(nodeList)) if freeNodes is not None else print('Get node by slurm')
    srunCode.append('--ntasks={}'.format(nTasks))
    srunCode.append('--nodes={}'.format(nodeNum))
    srunCode.append('--ntasks-per-node={}'.format(ntaskPerNode)) if envDistributed else print(
        'ntasks-per-node is 1')
    srunCode.append('--cpus-per-task=4')
    srunCode.append('--kill-on-bad-exit=1')
    srunCode.append('--mpi=pmi2')
    # srunCode.append(' --pty bash')
    srunCode.append(pyCode)

    srunCode = ' '.join(srunCode)

    os.system(srunCode)
    # else:
    #     os.system(pyCode)


if __name__ == '__main__':
    runDist()
