# API

## What is API?

- Short for Application Programming Interface
- A software intermediary that allows two applications to talk to each other

## How does API work?

- ### SOAP API 

  - Simple Object Access Portocal
  - Client and server exchange messages using XML
  - Less flexible
  - Popular in the past

- ### RPC API

  - Remote Procedure Call
  - Client completes a function (procedure) on server, and server sends the output back to client

- ### Websocket API

  - Modern web API
  - Use JSON objects to pass data
  - Support two-way communication between client apps and server
  - Server can send callback messages to connected clients
  - Effient than REST API

- ### REST API

  - Most popular and flexible
  - Client sends requests to the server as data
  - Server uses this client input to start internal functions and returns output data back to client

## What is API integration?

- Software component that automatically update data between clients and servers

## What are the different types of API?

- ### Private APIs

  - Internal to an enterprise and only used for connecting systems and data within the business

- ### Public APIs

  - Open to the public and may be used by anyone
  - May or be not be some authorization and cost associated with these types of APIs

- ### Partner APIs

  - Only accessible by authorized external developers to aid business-to-business partnerships

- ### Composite APIs

  - Combine two or more different APIs to address complex system requirements or behavior

## What is an API endpoint and why is it important?

- API endpoints are the final touchpoints in the API communication system
- Include server URLs, services, and other specific digital locations from where information is sent and received between systems
- API endpoints are critical to enterprises
- API endpoints make the system vulnerable to attack
- API monitoring is crucial for preventing misuse
- API endpoints can cause bottlenecks and affect system performance

## How to create an API?

- ### Plan the API

  - API specifications
  - Provide the blueprint for your API design
  - Better to think about different use cases in advance and ensure the API adheres to current API development standards

- ### Build the API

  - API designers prototype APIs using boilerplate code
  - Once the prototype is tested, developers can customize it to internal specifications

- ### Test the API

  - The same as software testing
  - Must be done to prevent bugs and defects
  - Testing tools can be used to strength test the API against cyber attacks

- ### Document the API

  - Acts as a guide to improve usability
  - Offer a range of functions and use cases tend to be more popular in a service-oriented architecture

- ### Market the API

  - API marketplaces exist for developers to buy and sell other APIs

# REST API

## What is REST API?

- Short for Representational State Transfer
- REST defines a set of functions like GET, PUT, DELETE, etc. that clients can use to access server data
- Clients and servers exchange data using HTTP
- Main feature of REST API: statelessness -- servers do not save client data between requests
- Response from server is plain data, without typical graphical rendering of a web page

## Benefits of REST API?

- ### Integration

  - APIs are used to integrate new applications with existing software systems
  - Increase development speed because each functionality doesn't have to be written from scratch
  - Can use APIs to leverage existing code

- ### Innovation

  - Entire industries can change with the arrival of a new app
  - Can make changes at the API level without having to re-write the whole code

- ### Expansion

  - APIs present a unique opportunity for businesses to meet their clients' needs across different platforms
  - Any business can give similar access to their internal databases by using free or paidAPIs

- ### Ease of maintenance

  - API acts as a gateway between two systems
  - Each system is obliged to make internal changes so that the API is not impacted
  - Any future code changes by one party do not impact the other party

## How can an API to be considered RESTful?

- A client-server architecture made up of clients, servers, and resources, with requests managed throught HTTP
- Stateless client-server communication, meaning no client information is stored between get requests and each request is separate and unconnected
- Cacheable data that streamlines client-server interactions
- A uniform interface between components so that information is transferred in a standard form
- A layered system that organizes each type of server involved the retrieval of requested information into hierarchies, invisible to the client
- Code on demand (optional): server can temporarily extend client and transfer logic to client

## How to secure a REST API?

- ### Authentication tokens

  - Used to authorize users to make the API call
  - Check that the users are who they claim to be and that they have access rights for that particular API call

- ### API keys

  - Verify the program or application making the API call
  - Identify the application and ensure it has the access rights required to make the particular API call
  - Not as secure as tokens but allow API monitoring in order to gather data on usage

## Design guideline

- ### URL design

  - verb + noun

    - GET (Read)

    - POST (Create)

    - PUT (Update)

    - PATCH (Partially update)

    - DELETE (Delete)

  - noun in plural form

    - GET /articles/2 better than GET /article/2

  - avoid multiple levels of URL

    - GET /authors/12?categories=2 instead of GET /authors/12/categories/2

- ### Status code

  - Every time client sent a request, server need to response with HTTP status code and data
  - HTTP status code is a number with 5 categories
    - 1xx: related to information
    - 2xx: successful operation
    - 3xx: redirect
    - 4xx: client error
    - 5xx: server error
  - API do not need 1xx status code
  - 2xx status code:
    - 200 OK: 		     successful operation
    - 201 Created:       create new content
    - 202 Accepted:     server received the request, but not processing, will process later
    - 204 No Content:  content not existed
  - 3xx status code:
    - 303 See Other:    refer to another URL
  - 4xx status code:
    - 400 Bad Request:                      server does not understand the request of client, did not do any process
    - 401 Unauthorized:                      user have not provide authentication credentials or did not pass authentication
    - 403 Forbidden:                           user passed the authentication, but have no permission to access the content
    - 404 Not Found:                           requested content not exist or unavailable
    - 405 Method Not Allowed:            user passed the authentication, but have no permission toward the HTTP method that he uses
    - 410 Gone:                                   requested content have been moved from this address, not available
    - 415 Unsupported Media Type:    the format client required is not supported
    - 422 Unprocessable Entity:          the appendix sent by client is not processable, cause request error
    - 429 Too Many Requests:            number of client requests exceed limit
  - 5xx status code:
    - 500 Internal Server Error:    request of client works but accident happens when server process it
    - 503 Service Unavailable:     server cannot process request, used for website maintenance status

- ### Response of server

  - Do not return plain text
  - Do not return 2xx status code when error occurs
  - Provide related URL



# Example of using Python to request for API

```python
# import requests and other packs
import requests
import json
import sys

reload(sys)
sys.setdefaultencoding("utf-8")

# define request sending function
def send_request(method, url, params=None, headers=None):
    method = str.upper(method)
    
    if method == "POST":
       return requests.post(url=url, data=params, headers=headers)
    elif method == "GET":
       return requests.get(url=url, params=params, headers=headers)
    else:
       return None

# global variables
method = "POST"
url = "https://api.apishop.net/common/air/getCityPM25Detail"
headers = None
params = {
   "apiKey" : "参数1",
   "city" : "参数2",
}

# get response
result = send_request(method=method, url=url, params=params, headers=headers)

if result:
   body = result.text
   response = json.loads(body)
   status_code = response["statusCode"]

   if (status_code == "000000"):
       print("request succeeded: %s" % (body,))
   else:
       print("request failed: %s" % (body,))
    
   else:
       print("sending request failed")
```



# Reference

https://www.mulesoft.com/resources/api/what-is-an-api

https://aws.amazon.com/what-is/api/?nc1=h_ls

https://www.redhat.com/en/topics/api/what-is-a-rest-api

https://www.cnblogs.com/bigsai/p/14099154.html

https://www.zhihu.com/question/35210451

https://www.infoq.cn/article/rest-introduction/

https://www.ruanyifeng.com/blog/2018/10/restful-api-best-practices.html

https://blog.csdn.net/weixin_43358075/article/details/90084945

https://blog.csdn.net/weixin_43358075/article/details/90084674?spm=1001.2014.3001.5502

https://www.apishop.net/#/api/example/?lang=Python&productID=94



