### Type Inference

##### Java 7 이전

제너릭 타입 파라미터를 선언과 생성시 중복해서 써야한다.

```java
Map<String, List<String>> employeeRecords = new HashMap<String, List<String>>();
List<Integer> primes = new ArrayList<Integer>();
```

##### Java 7 이후

생성자 영역의 타입 파라미터를 <>로 대체하여 생략할 수 있다.

```java
Map<String, List<String>> employeeRecords = new HashMap<>();
List<Integer> primes = new ArrayList<>(); 
```



### String in Switch

##### Java 7 이전

int, enum만 switch에 사용할 수 있다. 

##### java 7 이후

Switch문 내에서 문자열을 사용할 수 있다.

```java
switch (day) {
	case "NEW":
		System.out.println("Order is in NEW state");
		break;
	case "CANCELED":
		System.out.println("Order is Cancelled");
		break;
	case "REPLACE":
		System.out.println("Order is replaced successfully");
		break;
	case "FILLED":
		System.out.println("Order is filled");
		break;
	default:
	System.out.println("Invalid"); 
}
```



### Automatic Resource Management

##### Java 7 이전

DB Connecteion, File stream 등을 open했을 때 오류 발생시에도 정상적인 종료를 위해 finally 블럭안에서 close 처리를 해야한다.

```java
public static void main(String args[]) {
	FileInputStream fin = null;
	BufferedReader br = null;
	
	try {
		fin = new FileInputStream("info.xml");
		br = new BufferedReader(new InputStreamReader(fin));
		if (br.ready()) {
			String line1 = br.readLine();
			System.out.println(line1);
		}
	} catch (FileNotFoundException ex) {
		System.out.println("Info.xml is not found");
	} catch (IOException ex) {
		System.out.println("Can't read the file");
	} finally {
		try {
			if (fin != null)
				fin.close();
			if (br != null)
				br.close();
		} catch (IOException ie) {
			System.out.println("Failed to close files");
		}
	}
}  
```



##### Java 7 이후

**AutoClosable**, **Closeable** 인터페이스를 구현한 객체에 대하여 try 내에서 close를 해준다

```java
public static void main(String args[]) {
	try (FileInputStream fin = new FileInputStream("info.xml");
		BufferedReader br = new BufferedReader(new InputStreamReader(fin));) {
		if (br.ready()) {
			String line1 = br.readLine();
			System.out.println(line1);
		}
	} catch (FileNotFoundException ex) {
		System.out.println("Info.xml is not found");
	} catch (IOException ex) {
		System.out.println("Can't read the file");
	}
}
```



### Fork/Join Framework

Fork/Join Framework는 멀티프로세서의 성능을 이용할 수 있는 ExecutorService 인터페이스의 구현체이다.

반복적으로 작은 조각으로 작업을 나누어 수행 할 수 있게 설계 되었다.
어플리케이션의 성능을 향상 시키기 위해 가능한 모든 프로세서를 이용하기 위한 것

ExecutorServcie를 구현함으로써 Fork/Join Framework는 Thread Pool안의 Worker Thread에게 작업들을 분배한다.

Fork/Join Framework는 Produce-Consumer 알고리즘과는 매우 다른 work-stealing 알고리즘을 이용한다.
작업이 없는 Worker Thread는 아직 바쁜 다른 Thread의 작업을 가져올 수 있다.

Fork/Join Framework의 핵심은 AbstractExecutorService 클래스를 구현한 ForkJoinPool 클래스이다.
ForkJoinPool은 핵심적인 work-stealing 알고리즘을 구현하고 있다.
ForkJoinTask 프로세스들을 실행 할 수 있다.



### Underscore in Numeric literal

숫자형(정수,실수)에 _(underscore) 문자열을 사용하여 가독성을 향상 시킬 수 있다.

##### 사용 가능

```java
int billion = 1_000_000_000; // 10^9
long creditCardNumber = 1234_4567_8901_2345L; //16 digit number
long ssn = 777_99_8888L;
double pi = 3.1415_9265;
float pif = 3.14_15_92_65f;
```



##### 사용 불가능

```java
double pi = 3._1415_9265; // 소수점 뒤에 _ 붙일 경우
long creditcardNum = 1234_4567_8901_2345_L; // 숫자 끝에 _ 붙일 경우
long ssn = _777_99_8888L; // 숫자 시작에 _ 붙일 경우
```



### Catching Multiple Exception Type in Single Catch Block

catch 블럭에서 여러개의 Exception 처리가 가능하다 (Multi-Catch)

##### Java 7 이전

```java
try {
	//...... 
} catch(ClassNotFoundException ex) {
	ex.printStackTrace(); 
} catch(SQLException ex) {
	ex.printStackTrace(); 
}
```



##### Java 7 이후

```java
try {
	//......
} catch (ClassNotFoundException | SQLException ex) {
	ex.printStackTrace();
}
//////////////////////////////////////////////////////////////////////////////
//Multi-Catch 구문 사용시 Exception들이 하위클래스 관계라면 컴파일 에러가 발생한다 //
//////////////////////////////////////////////////////////////////////////////
try {
    //...... }
catch (FileNotFoundException | IOException ex) {
    ex.printStackTrace(); 
}
 
Alternatives in a multi-catch statement cannot be related by sub classing, it will throw error at compile time :
java.io.FileNotFoundException is a subclass of alternative java.io.IOException at Test.main
```



### Binary Literals with Prefix “0b”

숫자형에 **0B** 또는 **0b**를 앞에 붙임으로써 이진법 표현이 가능하다

8진법은 **0**

16진법은 **0X** 또는 **0x**

```java
int mask = 0b01010000101;
int binary = 0B0101_0000_1010_0010_1101_0000_1010_0010;    // _를 이용한 가독성 향상
```



### Java NIO 2.0

기본파일시스템에 접근도 가능하고 다양한 파일I/O 기능도 제공

- 파일을 이동
- 파일 복사
- 파일 삭제
- 파일속성이 Hidden인지 체크도 가능
- 심볼릭링크나 하드링크도 생성 가능
- 와일드카드를 사용한 파일검색도 가능
- 디렉토리의 변경사항을 감시하는 기능
- 등등..



### G1 Garbage Collector

- G1(Garbage First)
- 새로운 Garbage Collector가 추가
- G1 GC는 Garbage가 가장 많은 영역의 정리를 수행한다
- 메모리 집중적인 어플리케이션에 더 큰 Through put을 제공



### More Precise Rethrowing of Exception

- JDK7 이전 버젼에서는 catch 구문내에서 선언한 예외 유형만 밖으로 던질 수 있다
- JDK7에서는 catch 구문에서 선언한 예외를 밖으로 던질 수 있다

##### Java 7 이전

```java
public void obscure() throws Exception {
	try {
		new FileInputStream("abc.txt").read();
		new SimpleDateFormat("ddMMyyyy").parse("12-03-2014");
	} catch (Exception ex) {
		System.out.println("Caught exception: " + ex.getMessage());
		throw ex;
	}
}  
```



##### Java 7 이후

```java
public void precise() throws ParseException, IOException {
	try {
		new FileInputStream("abc.txt").read();
		new SimpleDateFormat("ddMMyyyy").parse("12-03-2014");
	} catch (Exception ex) {
		System.out.println("Caught exception: " + ex.getMessage());
		throw ex;
	}
}
```

