# Postman

## 介绍

### 创建简单GET请求

1. 选择请求方法GET
2. 在URL地址中填入测试接口地址
3. GET参数即可以直接填写在URL中，也可以点击Params，在参数表格中填写
4. 填好URL和参数后，点击’Send‘按钮即可发出一个GET请求

### 创建简单POST请求

1. 选择请求方法 POST

2. 在URL地址中填入测试接口地址

3. 在Body中，填写参数

4. 填好URl和参数后，点击’Send‘按钮即可发出一个POST请求

   

## HTTP请求与响应

### HTTP协议介绍

- Hyper Text Transfer Protocal
- 属于应用层的面向对象的传输协议
- 工作于客户端-服务端架构为上
- 特点：
  - 无连接 -- 限制每次连接只处理一个请求。服务器处理完客户的请求，并收到客户的应答后，即断开连接
  - 无状态 -- 每个请求都是独立的，不会自动记忆状态。聪明的人们，为了解决对话能保持状态，使用了session与cookie来解决

### HTTP请求的组成

- 起始行 -- 包含请求方法，请求地址
- 请求头Headers -- 格式如key:value，主要用于传递客户端的特定配置信息
- 请求体Body -- 请求体body: post方法存储参数的位置

### 用Postman发起API请求

1. 请求方法method
2. 请求地址URL
3. 请求头hearders
4. 请求体body

- ### Method

  - GET

  - HEAD

  - POST

  - PUT

  - DELETE

  - CONNECT

  - OPTIONS

  - TRACE

    

- ### URL

  - 点击按钮“Params”会展开参数编辑窗口，在窗口中填入的参数键值对，会自动拼接到URL参数中

  - 直接写在URL中的参数，也会自动以键值对的形式显示在编辑窗口

    

- ### 请求头Headers

  - Accept -- 可接受的相应内容类型 （Content-Types）
  - Accept-Charset -- 可接受的字符集
  - Accept-Encoding -- 可接受的响应内容的编码方式
  - Accept-Language -- 可接受的响应内容语言列表
  - Accept-Datetime -- 可接受的按照时间来表示的响应内容版本
  - Authorization -- 用于表示HTTP协议中需要认证资源的认证信息
  - Cache-Control -- 用来指定当前的请求/回复中的，是否使用缓存机制
  - Connection -- 客户端（浏览器）想要优先使用的连接类型
  - Cookie -- 由之前服务器通过Set-Cookie设置的一个HTTP协议Cookie
  - Content-Length -- 以8进制表示的请求体的长度
  - Content-MD5 -- 请求体的内容的二进制MD5散列值（数字签名），以Base64编码的结果
  - Content-Type -- 请求体的MIME类型（用于POST和PUT请求中）
  - Date -- 发送该消息的日期和时间（以RFC 7231中定义的"HTTP日期"格式来发送）
  - Expect -- 表示客户端要求服务器做出特定的行为
  - From -- 发起此请求的用户的邮件地址
  - Host -- 表示服务器的域名以及服务器所监听的端口号。如果所请求的端口是对应的服务的标准端口（80），则端口号可以省略
  - If-Match -- 仅当客户端提供的实体与服务器上对应的实体相匹配时，才进行对应的操作。主要用于像PUT这样的方法中，仅当从用户上次更新某个资源后，该资源未被修改的情况下，才更新该资源
  - If-Modified-Since -- 允许在对应的资源未被修改的情况下返回304未修改
  - If-None-Match -- 允许在对应的内容未被修改的情况下返回304未修改，参考超文本传输协议的实体标记
  - If-Range -- 如果该实体未被修改过，则向返回所缺少的哪一个或多个部分。否则，返回整个新的实体
  - If-Unmodified-Since -- 仅当该实体自某个特定的时间以来未被修改的情况下，才发送回应
  - Max-Forwards -- 限制该消息可被代理及网关转发的次数
  - Origin -- 发起一个针对跨域资源共享的请求（该请求要求服务器在响应中加入一个Access-Control-Allow-Origin的消息头，表示访问控制所允许的来源）
  - Pragma -- 与具体的实现相关，这些字段可能在请求/回应链中的任何时候产生
  - Proxy-Authorization -- 用于向代理进行认证的认证信息
  - Range -- 表示请求某个实体的一部分，字节偏移以0开始
  - Referer -- 表示浏览器所访问的前一个页面，可以认为是之前访问页面的链接将浏览器带到了当前页面
  - TE -- 浏览器预期接受的传输时的编码方式：可使用回应协议头Transfer-Encoding中的值（还可以使用“trailers”表示数据传输时的分块方式）用来表示浏览器希望在最后一个大小为0的块之后还接收到一些额外的字段
  - User-Agent -- 浏览器的身份识别字符串
  - Upgrade -- 要求服务器升级到一个高版本协议
  - Via -- 告诉服务器，这个请求是由哪些代理发出的
  - Warning -- 一个一般性的警告，表示在实体内容体中可能存在的错误

- ### Cookies

  - Cookie不能跨域访问
  - 在设置cookie时，需要先指定域名，然后设置cookie内容

- ### Body

  - 在POST方法中，参数通常设置在Body
  - 在Body参数的格式上，有四种：“form-data”，“urlencoded”，“raw”，“binary”
  - “form-data”类型
    - 是默认的web表单数据传输的编码类型
    - 模拟了在网站上填写表单，并提交他
    - 既可以上传键值对，也可以上传文件
  - “x-www-form-urlencoded”类型
    - 只能上传键值对
    - 会将表单内的数据转换为键值对，并自动对参数值进行urlencode编码
  - “raw”类型
    - 可以上传任意格式的文本
    - 可以上传text，json，xml，html等
  - “binary”类型
    - 只可以上传二进制数据
    - 通常用来上传文件

### HTTP响应的组成

- 状态行
- 响应头Headers
- 响应体Body

### 响应头Headers

- Access-Control-Allow-Origin -- 指定哪些网站可以跨域资源共享
- Accept-Patch -- 指定服务器所支持的文档补丁格式
- Accept-Rangers -- 服务器所支持的内容范围
- Age -- 响应对象在代理缓存中存在的时间，以秒为单位
- Allow -- 对于特定资源的有效动作
- Cache-Control -- 通知从服务器到客户端的所有缓存机制，表示他们是否可以缓存这个对象及缓存的有效时间，以秒为单位
- Connection -- 针对该连接所预期的选项
- Content-Disposition -- 对已知MIME类型资源的描述，浏览器可以根据这个响应头决定是对返回资源的动作，如：将其下载或是打开
- Content-Encoding -- 响应资源所使用的编码类型
- Content-Language -- 响应内容所使用的语言
- Content-Length -- 响应消息体的长度，用8进制字节表示
- Content-Location -- 所返回的数据的一个侯选位置
- Content-MD5 -- 响应内容的二进制MD5散列值，以Base64方式编码
- Content-Range -- 如果是响应部分消息，表示属于完整消息的哪个部分
- Content-Type -- 当前内容的MIME类型
- Date -- 此条消息被发送时的日期和时间（以RFC 7231中定义的“HTTP日期”格式来表示）
- ETag -- 对于某个资源的某个特定版本的一个标识符，通常是一个消息散列
- Expires -- 指定一个日期/时间，超过该时间则认为此回应已经过期
- Last-Modified -- 所请求的对象的最后修改日期（按照RFC 7231中定义的“超文本传输协议日期”格式来表示）
- Link -- 用来表示与另一个资源之间的类型关系，此类型关系是在RFC 5988中定义
- Location -- 用于在进行重定向，或在创建了某个新资源时使用
- P3P -- P3P策略相关设置
- Pragma -- 与具体的实现相关，这些响应头可能在请求/回应链中的不同时候产生不同的效果
- Proxy-Authenticate -- 要求在访问代理时提供身份认证信息
- Public-Key-Pins -- 用于防止中间攻击，声明网站认证中传输层安全协议的证书散列值
- Refresh -- 用于重定向，或者当一个新的资源被创建时。默认会在5秒后刷新重定向
- Retry-After -- 如果某个实体临时不可用，那么此协议头用于告知客户端稍后重试。其值可以是一个特定的时间段（以秒为单位）或一个超文本传输协议日期
- Server -- 服务器的名称
- Set-Cookie -- 设置HTTP cookie
- Status -- 通用网关接口的响应头字段，用来说明当前HTTP连接的响应状态
- Trailer -- Trailer用户说明传输中分块编码的编码信息
- Transfer-Encoding -- 用表示实体传输给用户的编码形式。包括：chunked，compress，deflate，gzip，identity
- Upgrade -- 要求客户端升级到另一个高版本协议
- Vary -- 告知下游的代理服务器，应当如何对以后的请求协议头进行匹配，以决定是否可使用已缓存的响应内容，而不是重新从原服务器请求新的内容
- Via -- 告知代理服务器的客户端，当前响应是通过什么途径发送的
- Warning -- 一般性警告，告知在实体内容体中可能存在错误
- WWW-Authenticate -- 表示在请求获取这个实体时应当使用的认证模式



## Collection管理与运行

### 一般项目结构

- Collection项目（测试集）
  - Folder模块
    - request请求

### Collection介绍

- 测试集
- 用于组织和管理接口用例相关信息
- 可执行文件
- 是API和Postman内置工具的核心
- 是内置工具的执行的开始，包括：MOCK，文档，测试，监控，发布
- 测试集名称 -- 可以采用项目名称
- 描述文档 -- 描述项目的信息
- 前置脚本 -- 在这个测试集下的每个接口，请求发送之前执行的脚本，一般用于参数构造处理
- 断言测试 -- 在这个测试集下的每个接口响应后，都会执行的断言
- 变量设置 -- 在这个测试集下的所有接口请求，都可以共享的变量

### 分享Collection

- 通过分享URL
- 通过共享项目
- 通过导出Collection文件

### 运行Collection

- 通过app中的Runner按钮运行
- 通过Newman命令行工具运行
- 通过Collection文件运行
- 通过Collection分享URL运行



## 变量详解

### 变量的目的

1. 一处设置，多出调用，一处修改，全部生效
2. 方便数据的关联与同步

### 变量的作用范围

- 优先级从大到小：
  - data
  - local
  - environment
  - collection
  - global

### 

## 初级脚本使用

### Scripts脚本介绍

- Postman中，有个叫“Sandbox”的运行环境，是一个JavaScript执行环境
- Postman接口测试过程中，有两个位置我们可以添加自己的脚本：
  - pre-request script -- 一般用于数据初始化
  - test script -- 用于断言（解析响应内容，与预期值进行对比）

### Script代码片段介绍

- 环境变量

  - ```
    pm.environment.set("timestampsParam", new Data());     // 设置环境变量
    ```

  - ```
    pm.environment.get("timestampsParam");     // 获取环境变量
    ```

  - ```
    pm.environment.unset("timestampsParam");     // 删除环境变量
    ```

- 全局变量

  - ```
    pm.globals.set("variable_key", "variable_value");    // 设置全局变量
    ```

  - ```
    pm.globals.get("variable_key");     // 获取全局变量
    ```

  - ```
    pm.globals.unset("variable_key");     // 删除全局变量
    ```

- 变量

  - ```
    pm.variables.get("variable_key");     // 在全局变量和活动环境中搜索变量
    ```

- Data变量

  - 只有在Collection运行中使用，所有的Data变量会存储在data对象内，假如存在变量value，则data.value可获得当前迭代时的value变量值

  - ```
    var jsonData = JSON.parse(responseBody);
    tests['Response has data value'] = jsonData.form.foo === data.value
    ```

- 将JSON对象（对象或数组）转JSON字符串

  - ```
    var array = [1, 2, 3, 4];
    var jsonString = JSON.stringify(array);
    ```

- 将JSON字符串转JSON对象Object

  - ```
    var array = "[1, 2, 3, 4]";
    var jsonObject = JSON.parse(array);
    ```

- XML字符串转JSON对象Object

  - ```
    var jsonObject = xml2Json(XMLString);
    ```

- 请求属性

  - request：对象类型，只读

  - ```
    console.log(request.data);     // request中的form-data对象
    console.log(request.headers);     // request中的headers对象
    console.log(request.method);     // request中的meth字符串
    console.log(request.url);     // request中的url字符串
    ```

- 响应属性

  - 只能在Tests Script中使用

  - responseBody：响应body的原始字符串

  - responseTime：响应时间，毫秒

  - responseCode：响应码对象类型

  - tests：检查点对象类型，需要我们自己填充检查点

  - ```
    console.log(responseBody);     // body字符串
    console.log(responseTime);     // 响应时间
    console.log(responseCode);     // 响应码对象
    tests["title"]=responseTime<1000;     // 响应时间小于1000毫秒
    console.log(tests)     // 检查点对象
    ```

- 将响应内容转成JSON对象

  - ```
    pm.response.json()
    ```

- 将响应内容转成text字符串

  - ```
    pm.response.text()
    ```

- 发送异步HTTP请求

  - ```
    pm.sendRequest("http://postman-echo.com/get", function(err, response) {
    	console.log(response.json());
    });
    ```

- 断言（检查点）

  - ```
    tests["响应时间小于1000毫秒"]=responseTime<1000;     // 检查点结果是一个表达式
    ```

  - ```
    pm.test("检查点名称", function() {
    	pm.expect(...);
    });     // 检查点结果是一个回调函数
    ```

- 响应Body为字符串时，断言包含特定字符串

  - ```
    pm.test("Body matches string", fucntion() {
    	pm.expect(pm.response.text()).to.include("string_you_want_to_search");
    });
    ```

- 响应Body为字符串时，断言完全匹配特定字符串

  - ```
    pm.test("Body is correct", function() {
    	pm.response.to.have.body("response_body_string");
    });
    ```

- 响应Body为JSON格式时，断言具体值

  - ```
    pm.test("Your test name", function() {
    	var jsonData = pm.response.json();
    	pm.expect(jsonData.value).to.eql(100);
    });
    ```

- 断言响应时间

  - ```
    pm.test("Response time is less than 200ms", function() {
    	pm.expect(pm.response.responseTime).to.be.below(200);
    });
    ```

- 断言响应码

  - ```
    pm.test("Status code is 200", function() {
    	pm.response.to.have.status(200);
    });
    ```

- 全局函数pm详解

  - ```
    pm.sendRequest:Function     // 异步发送HTTP/HTTPS请求
    ```

  - ```
    pm.globals.has(variableName:String):Function -> Boolean     // 判断全局变量是否存在
    ```

  - ```
    pm.globals.get(variableName:String):Function -> *     // 获取全局变量
    ```

  - ```
    pm.globals.set(variableName:String, variableValue:String):Function     // 设置全局变量
    ```

  - ```
    pm.globals.unset(variableName:String):Function     // 删除某个全局变量
    ```

  - ```
    pm.globals.clear():Function     // 清空所有全局变量
    ```

  - ```
    pm.environment.has(variableName:String):Function -> Boolean     // 判断环境变量是否存在
    ```

  - ```
    pm.environment.get(variableName:String):Function -> *     // 获取环境变量
    ```

  - ```
    pm.environment.set(variableName:String, variableValue:String):Function     // 设置环境变量
    ```

  - ```
    pm.environment.unset(variableName:String):Function     // 删除某环境变量
    ```

  - ```
    pm.environment.clear():Function     // 清空环境变量
    ```

  - ```
    pm.variables.get(variableName:String):function -> *     // 获取某变量
    ```

  - ```
    pm.request.url:URL     // 返回请求URL
    ```

  - ```
    pm.request.headers:HeaderList     // 返回请求header列表
    ```

  - ```
    pm.response.code:Number     // 返回响应码
    ```

  - ```
    pm.response.reason():Function -> String     // 返回响应消息
    ```

  - ```
    pm.response.headers:HeaderList     // 返回响应头headers列表
    ```

  - ```
    pm.response.responseTime:Number     // 返回响应时间，毫秒
    ```

  - ```
    pm.response.text():Function -> Stiring     // 返回响应Body字符串
    ```

  - ```
    pm.response.json():Function -> Object     // 返回响应Body json对象
    ```

  - ```
    pm.cookies.has(cookieName:String):Function -> Boolean     // 判断是否存在某cookie
    ```

  - ```
    pm.cookies.get(cookieName:String):Function -> String     // 获取某cookie
    ```

  - ```
    pm.test(testName:String, specFunction:Function):Function     // 编写断言
    ```

  - ```
    pm.expect(assertion:*):Function -> Assertion     // 通用的断言函数
    ```

    

# Reference

https://www.kancloud.cn/guanfuchang/tools/691823