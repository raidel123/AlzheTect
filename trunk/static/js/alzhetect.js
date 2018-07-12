$.inputjq = function(){
  //alert(id);
  var newDiv1 = $('<form id="classifier_form" class="w3-container"> <div class="row"> {% for field in patient_input_fields.field_name %} <div class="col-lg-6 col-md-6"><p><label id="{{field}}_label">{{field[0]}}</label><input id="{{field}}_input." class="w3-input" type="text"></p></div>{% endfor %}</div></form>');
  var newDiv2 = $('<div id="classifier_form" class="row"><form action = "/uploader/" method = "POST" enctype = "multipart/form-data"><input type = "file" name = "file" /><input type = "submit"/></form></div>');

  $('#classifier_input').append(newDiv1);
  $('#classifier_input').append(newDiv2);
};

function changeInput(self, show_id){
	// alert(self.value);

  var raw_input = document.getElementById('raw_input');
  var file_input = document.getElementById('file_input');
  if(show_id == 'raw_input'){
    file_input.style.display = "none";
    raw_input.style.display = "block";
  } else {
    file_input.style.display = "block";
    raw_input.style.display = "none";
  }

  // var parent = document.getElementById("classifier_input");
  // var child = document.getElementById("classifier_form");
  // parent.removeChild(child);

  // $.inputjq()

  // $('#classifier_input').append(newDiv1);
  // $('#classifier_input').append(newDiv2);

}
