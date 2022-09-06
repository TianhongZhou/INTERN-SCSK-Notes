# Docker

## 什么是Docker？

- PaaS 提供商 dotCloud基于容器技术的轻量级虚拟化解决方案
- 基于Google公司推出的Go 语言实现 
- 遵从Apache 2.0 协议
- Docker是一种容器化技术的落地
- 容器作为一类操作系统层面的虚拟化技术，其目标是在单一Linux主机交付多套隔离性环境，容器共享同一套主机操作系统内核



## 什么是容器技术？

- 容器是一种沙盒技术，主要目的是为了将应用运行在其中，与外界隔离；及方便这个沙盒可以被转移到其它宿主机器
- 本质上，它是一个特殊的进程
- 通过名称空间（Namespace）、控制组（Control groups）、切根（chroot）技术把资源、文件、设备、状态和配置划分到一个独立的空间

### 原理

- 为了实现容器进程对外界的隔离，容器底层主要运用了名称空间（Namespaces）、控制组（Control groups）和切根（chroot）
- 名称空间（Namespaces）
  - 每个运行的容器都有自己的名称空间
  - 这是Linux操作系统默认提供的API，包括
    - PID Namespace
      - 不同容器就是通过pid名字空间隔离开的，不同名字空间中可以有相同的pid
    - Mount Namespace
      - mount允许不同名称空间的进程看到的文件结构不同，因此不同名称空间中的进程所看到的文件目录就被隔离了
      - 每个名称空间中的容器在/proc/mounts的信息只包含当前名称的挂载点
    - IPC Namespace
      - 容器中进程交互还是采用Linux常见的进程交互方法（interprocess communication -IPC），包括信号量、消息队列和共享内存等
    - Network Namespace
      - 网络隔离是通过Net实现，每个Net有独立的网络设备，IP地址，路由表，/proc/net目录
      - 这样每个容器的网络就能隔离开来
    - UTS Namespace
      - UTS（UNIX Time-sharing System）允许每个容器拥有独立的hostname和domain name，使其在网络上可以被视作一个独立的节点而非主机上的一个进程
    - User Namespace
      - 每个容器可以有不同的用户和组id，也就是说可以在容器内用容器内部的用户执行程序而非主机上的用户
  - 控制组（Control groups）
    - 是Linux内核提供的一种可以限制、记录、隔离进程组的物理资源机制
    - 采用Cgroup技术对容器进行资源限制，防止某个容器把宿主机资源全部用完导致其它容器也宕掉
    - 在Linux的/sys/fs/cgroup目录中，有cpu、memory、devices、net_cls等子目录，可以根据需要修改相应的配置文件来设置某个进程ID对物理资源的最大使用率
  - 切根（chroot）
    - 改变一个程序运行时参考的根目录位置，让不同容器在不同的虚拟根目录下工作，从而相互不直接影响

### 容器与虚拟机

- 虚拟机通常包括整个操作系统和应用程序，里面运行的是一个真实的操作系统

- 本质上虚拟机是Hypervisor虚拟化出来的硬件上安装不同的操作系统，而容器是宿主机上运行的不同进程

- 从用户体验上来看，虚拟机是重量级的，占用物理资源多，启动时间长

- 容器则占用物理资源少，启动迅速

- 相对地，虚拟机隔离的更彻底，容器则要差一些

- 容器为应用程序提供了隔离的运行空间：每个容器内都包含一个独享的完整用户环境空间，并且一个容器内的变动不会影响其他容器的运行环境

- 容器技术使用了namespaces来进行空间隔离，通过文件系统的挂载点来决定容器可以访问哪些文件，通过cgroups来确定每个容器可以利用多少资源

- 此外容器之间共享同一个系统内核，这样当同一个库被多个容器使用时，内存的使用效率会得到提升

- 虚拟层为用户提供了一个完整的虚拟机：包括内核在内的一个完整的系统镜像

- CPU虚拟化技术可以为每个用户提供一个独享且和其他用户隔离的系统环境，虚拟层可以为每个用户分配虚拟化后的CPU、内存和IO设备资源

  

## Docker

### 原理

- 把linux的cgroup、namespace等容器底层技术进行封装抽象，为用户提供了创建和管理容器的便捷界面（命令行和API）

### 特点

- 一次构建，可以运行在任何地方
- 跨平台和强一致性

### 三组件

- 镜像
  - 镜像可以用来创建Docker容器
  - 一个镜像可以包含一个完整的操作系统环境和用户需要的其它应用程序，docker的镜像是只可读的，一个镜像可以创建多个容器
- 容器
  - 容器是镜像创建的实例
  - 它可以被启动、开始、停止、删除。每个容器都是相互隔离的、保证安全的平台
- 仓库
  - 仓库是集中存放镜像文件的场所
  - 每个仓库中又包含了多个镜像，每个镜像有不同的标签

### 架构

- Docker的基础架构是客户端-服务器(client-server)模式
- 在Docker中的主要组件有守护程序进程（daemon process）[服务器（service）,一种长期运行的程序]，命令行界面客户端（command line interface client， CLI client）和指定程序与守护进程通信并指示其操作的接口REST API
- CLI 使用接口docker REST API通过脚本或者直接CLI命令控制docker守护进程或者与docker守护进程进行交互
- 守护进程(daemon)创建和管理docker对象，比如镜像(images)，容器（contains），网络（network）和数据卷(data volumes)
- Docker客户端和Docker守护进程可以在同一系统上运行，也可以将Docker客户端连接到远程的Docker守护进程
- Docker客户端和Docker守护进程在UNIX套接字或者网络接口上使用REST API进行通话，Docker守护进程用来完成docker容器的构建，运行和分发等工作
- docker守护进程
  - 侦听docker API的并且管理docker对象，例如图像，容器，网路和数据卷
  - 守护进程也可以与其他的守护进程进行通信来管理docker服务
- docker客户端
  - 是docker用户与Docker交互的主要方式
  - 客户端将命令发送到守护进程，守护进程执行相应的命令
  - Docker客户端可以与多个守护进程进行通信
- docker仓库
  - 用来储存docker仓库
  - Docker hub是一个任何人都能够使用的公共仓库
- docker对象
  - 镜像是一个带有创建Docker容器的说明的只读模板
  - 用户可以docker仓库中他人已经创建好的镜像，也可以创建Dockerfile文件来编写创建镜像的指令
  - Dockerfile中的每条指令都会在镜像中创建一层
  - 当用户更改Dockerfile并且重新创建镜像的时候，只有更改的层需要重建，其它层保持不变
  - 通常情况下，一个镜像可以基于另一个镜像进行构建。假设用户现在拥有一个GPU版本的pytorch环境基础镜像，然而项目还需要一些额外的库文件，比如opencv库文件，那么就可以基于基础镜像，再安装相应的opencv库文件 ，就可以构造项目所需要的镜像了
  - 容器是一个镜像的可运行实例
  - 用户可以通过API或者命令行界面创建，启动，停止，移除和删除一个容器
  - 默认情况下，容器与容器以及宿主机之间的隔离度相对较好，用户可以控制容器的网络，存储和其他基础子系统与宿主机的隔离程度

### 常用命令

```
# 镜像有关命令
docker image pull    # 拉取镜像
docker images 	# 查看镜像
docker rmi  image-id/镜像名字  # 删除某个镜像
docker rmi $(docker images | grep -v RESPOSITORY | awk '{print $3}') # 删除所有镜像
docker search 镜像名字  # 搜索某个镜像
docker build -t 镜像名称：版本   .    # 构建镜像，注意后面的 .

# docker命令
docker -v  # 查看docker版本
docker info # 查看docker系统信息

# 容器有关命令
docker ps -a  # 查看所有容器列表
docker ps -a -n=10 # 查看10个容器
docker inspect 容器ID  # 查看某个容器的信息
docker rm 容器ID # 删除某个容器
docker rm $(docker ps -a)  # 删除所有容器
docker stop 容器ID # 关闭运行中的某个容器
docker start 容器ID # 启动某个容器但是不进入
docker start -i 容器ID # 启动并进入某个容器
docker restart 容器ID # 重启某个容器
docker attach 容器ID # 进入一个运行中的容器
docker run -it 镜像名称：版本 # 启动容器并且以交互式进入容器
```

