const requests = [];
function addRequest(){
  requests.push([document.getElementById("reg"),document.getElementById("name"),document.getElementById("req")]);
  alert("Meeting Request Generated Successfully");
}

function validate(){
  if(document.getElementById("pwd").value!="technology@123"){
    alert("Wrong Password");
  }
  else{
    var e = "<hr/>";
    for (var y=0; y<requests.length; y++)
    {
     e += "Request " + (y+1) + " = " + requests[y] + "<br/>";
    }
    document.getElementById("result").innerHTML = e;
    var x = document.createElement("INPUT");
    x.setAttribute("type", "time");
    x.style.height="30px";
    document.getElementById("result").innerHTML+="<br>";
    document.getElementById("result").innerHTML+="Schedule time for the meeting 1 : ";
    document.getElementById("result").appendChild(x);
  }
}
