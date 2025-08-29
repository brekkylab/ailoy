pub trait Normalize {
    fn normalized(&self) -> Self;
}

#[allow(unused_imports)]
impl Normalize for Vec<f32> {
    fn normalized(&self) -> Self {
        let magnitude = self.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude == 0.0 {
            return self.clone();
        }
        self.iter().map(|x| x / magnitude).collect()
    }
}
